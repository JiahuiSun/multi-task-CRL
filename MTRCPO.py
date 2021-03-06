import time
import pickle
import os
from tqdm import tqdm
import torch as th
import torch.nn.functional as F
from numba import njit
import numpy as np

from task_scheduler import CircularList


@njit
def _gae_return(
    v_s: np.ndarray,
    v_s_: np.ndarray,
    rew: np.ndarray,
    end_flag: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> np.ndarray:
    returns = np.zeros(rew.shape)
    delta = rew + v_s_ * gamma - v_s
    m = (1.0 - end_flag) * (gamma * gae_lambda)
    gae = 0.0
    for i in range(len(rew) - 1, -1, -1):
        gae = delta[i] + m[i] * gae
        returns[i] = gae
    return returns


@njit
def _discount_cumsum(rew, end_flag, gamma):
    returns = np.zeros(rew.shape)
    m = (1.0 - end_flag) * gamma
    gae = 0.0
    for i in range(len(rew) - 1, -1, -1):
        gae = rew[i] + m[i] * gae
        returns[i] = gae
    return returns


def gaussian_kl(mu0, log_std0, mu1, log_std1):
    """Returns average kl divergence between two batches of dists"""
    var0, var1 = np.exp(2 * log_std0), np.exp(2 * log_std1)
    pre_sum = 0.5*(((mu1 - mu0)**2 + var0)/(var1 + 1e-8) - 1) + log_std1 - log_std0
    all_kls = np.sum(pre_sum, axis=1)
    return np.mean(all_kls)


class MTRCPO:
    def __init__(
        self,
        env,
        task_sche,
        state_shape,
        action_shape,
        actor,
        critic,
        cost_critic,
        penalty,
        actor_optim,
        critic_optim,
        penalty_optim,
        dist_fn,
        save_freq=100,
        writer=None,
        device='cpu',
        # 接下来的参数可调
        n_epoch=1000,
        episode_per_proc=10,
        value_clip=False,
        clip=0.2,
        kl_stop=False,
        kl_margin=1.2,
        target_kl=0.01,
        repeat_per_collect=10,
        lr_actor=3e-4,
        lr_critic=1e-3,
        lr_penalty=5e-2,
        gamma=0.99,
        lam=0.97,
        cost_gamma=0.99, 
        cost_lam=0.97,
        norm_adv=False,
        recompute_adv=False,
        max_grad_norm=0.5
    ) -> None:
        self.env = env
        self.task_sche = task_sche
        self.state_dim = np.prod(state_shape)
        self.act_dim = np.prod(action_shape)
        self.actor = actor
        self.critic = critic
        self.cost_critic = cost_critic
        self.penalty = penalty
        self.actor_optim = actor_optim
        self.critic_optim = critic_optim
        self.penalty_optim = penalty_optim
        self.dist_fn = dist_fn

        self.n_epoch = n_epoch
        self.episode_per_proc = episode_per_proc
        self.value_clip = value_clip
        self.clip = clip
        self.kl_stop = kl_stop
        self.kl_margin = kl_margin
        self.target_kl = target_kl
        self.repeat_per_collect = repeat_per_collect
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.lr_penalty = lr_penalty
        self.gamma = gamma
        self.lam = lam
        self.cost_gamma = cost_gamma
        self.cost_lam = cost_lam
        self.norm_adv = norm_adv
        self.recompute_adv = recompute_adv
        self.max_grad_norm = max_grad_norm
        self.save_freq = save_freq
        self.writer = writer
        self.device = device

        self.log_dir = writer.log_dir
        self.nproc = len(env)
        self.step_per_episode = 1000
        self.step_per_task = self.step_per_episode * self.episode_per_proc
 
    def learn(self):
        st_time = time.time()
        all_epoch_cost = 0
        for epoch in tqdm(range(self.n_epoch)):
            st1 = time.time()
            sub_task_list = self.task_sche.subset()
            buffer = self.rollout(sub_task_list=sub_task_list)
            st2 = time.time()

            # training
            buffer = self.compute_gae(buffer)
            penalty = F.softplus(self.penalty.detach())
            policy_update_flag = 1
            for repeat in range(self.repeat_per_collect):
                if self.recompute_adv and repeat > 0:
                    buffer = self.compute_gae(buffer)

                all_task_actor_loss, all_task_critic_loss, all_task_kl = 0, 0, 0
                for param, data in buffer.items():
                    value, cost_value, curr_log_probs, mu, sigma = self.evaluate(data['obs'], np.array(param), data['act'])

                    # Calculate loss for critic
                    target = th.tensor(data['target'], dtype=th.float32, device=self.device)
                    cost_target = th.tensor(data['cost_target'], dtype=th.float32, device=self.device)
                    # NOTE: 只有不recompute adv时，value clip才有意义
                    if self.value_clip:
                        vs = th.tensor(data['vs'], dtype=th.float32, device=self.device)
                        v_clip = vs + (value - vs).clamp(-self.clip, self.clip)
                        vf1 = (target - value).pow(2)
                        vf2 = (target - v_clip).pow(2)
                        critic_loss = th.max(vf1, vf2).mean()

                        vs = th.tensor(data['cost_vs'], dtype=th.float32, device=self.device)
                        v_clip = vs + (cost_value - vs).clamp(-self.clip, self.clip)
                        vf1 = (cost_target - cost_value).pow(2)
                        vf2 = (cost_target - v_clip).pow(2)
                        cost_critic_loss = th.max(vf1, vf2).mean()
                    else:
                        critic_loss = (target - value).pow(2).mean()
                        cost_critic_loss = (cost_target - cost_value).pow(2).mean()
                    total_critic_loss = critic_loss + cost_critic_loss
                    all_task_critic_loss += 1 / len(sub_task_list) * total_critic_loss

                    # Calculate loss for actor
                    adv = th.tensor(data['adv'], dtype=th.float32, device=self.device)
                    cost_adv = th.tensor(data['cost_adv'], dtype=th.float32, device=self.device)
                    # NOTE: 不管是否recompute adv，old_log_probs都不变，都是原来执行时产生的值
                    old_log_probs = th.tensor(data['log_prob'], dtype=th.float32, device=self.device)
                    if self.norm_adv:
                        adv = (adv - adv.mean()) / adv.std()
                        cost_adv = cost_adv - cost_adv.mean()
                    ratios = th.exp(curr_log_probs - old_log_probs)
                    surr1 = ratios * adv
                    surr2 = th.clamp(ratios, 1 - self.clip, 1 + self.clip) * adv
                    surr_rew_loss = -th.min(surr1, surr2).mean()
                    surr_cost_loss = -(ratios * cost_adv).mean()
                    total_actor_loss = surr_rew_loss - penalty * surr_cost_loss
                    total_actor_loss /= (1 + penalty)  # why?
                    all_task_actor_loss += 1 / len(sub_task_list) * total_actor_loss

                    # Calculate kl
                    d_kl = gaussian_kl(mu, sigma, data['mu'], data['sigma'])
                    all_task_kl += 1 / len(sub_task_list) * d_kl

                self.critic_optim.zero_grad()
                all_task_critic_loss.backward()
                if self.max_grad_norm:  # clip large gradient
                    th.nn.utils.clip_grad_norm_(
                        list(self.critic.parameters())+list(self.cost_critic.parameters()), 
                        max_norm=self.max_grad_norm
                    )
                self.critic_optim.step()

                if policy_update_flag:
                    self.actor_optim.zero_grad()
                    all_task_actor_loss.backward()
                    if self.max_grad_norm:  # clip large gradient
                        th.nn.utils.clip_grad_norm_(
                            self.actor.parameters(), 
                            max_norm=self.max_grad_norm
                        )
                    self.actor_optim.step()

                    if self.kl_stop and all_task_kl > self.kl_margin * self.target_kl:
                        policy_update_flag = 0
                        print(f'Early stopping at step {repeat} due to reaching max kl.')

            # update penalty
            avg_cumu_cost = np.mean([data['avg_cumu_cost'] for _, data in buffer.items()])
            avg_threshold = np.mean([task[-1] for task in sub_task_list])
            penalty_loss = -self.penalty * (avg_cumu_cost - avg_threshold)
            self.penalty_optim.zero_grad()
            penalty_loss.backward()
            self.penalty_optim.step()

            # log everything
            end_time = time.time()
            avg_cumu_rew = np.mean([data['avg_cumu_rew'] for _, data in buffer.items()])
            all_epoch_cost += avg_cumu_cost
            cost_rate = all_epoch_cost / ((epoch+1)*self.step_per_episode)
            self.writer.add_scalar('param/avg_threshold', avg_threshold, epoch)
            self.writer.add_scalar('param/penalty', penalty, epoch)
            self.writer.add_scalar('metric/avg_cumu_rew', avg_cumu_rew, epoch)
            self.writer.add_scalar('metric/avg_cumu_cost', avg_cumu_cost, epoch)
            self.writer.add_scalar('metric/cost_rate', cost_rate, epoch)
            self.writer.add_scalar('loss/all_task_actor_loss', all_task_actor_loss.item(), epoch)
            self.writer.add_scalar('loss/all_task_critic_loss', all_task_critic_loss.item(), epoch)
            self.writer.add_scalar('time/rollout', st2-st1, epoch)
            self.writer.add_scalar('time/training', end_time-st2, epoch)
            self.writer.add_scalar('time/elapsed', time.time()-st_time, epoch)
            
            # save everything
            if (epoch+1) % self.save_freq == 0:
                actor_path = os.path.join(self.log_dir, f'actor-{epoch}.pth')
                critic_path = os.path.join(self.log_dir, f'critic-{epoch}.pth')
                cost_critic_path = os.path.join(self.log_dir, f'cost_critic-{epoch}.pth')
                th.save(self.actor.state_dict(), actor_path)
                th.save(self.critic.state_dict(), critic_path)
                th.save(self.cost_critic.state_dict(), cost_critic_path)

                # 保存obs的normalization参数
                norm_param_path = os.path.join(self.log_dir, f'obs_rms-{epoch}.pkl')
                with open(norm_param_path, 'wb') as f:
                    pickle.dump(self.env.obs_rms, f)
        self.env.close()

    def rollout(self, sub_task_list=[]):
        """
        1. 共sample n_task x episode_per_task条episode，每条episode有1000个step；
            - 逻辑：nproc个task并行采样episode_per_task条episode，换下一批task
        2. 统计平均累积reward和平均累积cost
        """
        buffer = {
            tuple(param): {
                'obs': [], 
                'act': [], 
                'mu': [],
                'sigma': [],
                'log_prob': [], 
                'rew': [], 
                'cost': [],
                'done': [],
                'obs_next': []
            } 
            for param in sub_task_list
        }

        def track_reset(step_per_task): 
            nproc_track = {
                'obs': np.zeros((step_per_task, self.nproc, self.state_dim), dtype=np.float32),
                'act': np.zeros((step_per_task, self.nproc, self.act_dim), dtype=np.float32),
                'mu': np.zeros((step_per_task, self.nproc, self.act_dim), dtype=np.float32),
                'sigma': np.zeros((step_per_task, self.nproc, self.act_dim), dtype=np.float32),
                'log_prob': np.zeros((step_per_task, self.nproc), dtype=np.float32),
                'rew': np.zeros((step_per_task, self.nproc), dtype=np.float32),
                'cost': np.zeros((step_per_task, self.nproc), dtype=np.float32),
                'done': np.zeros((step_per_task, self.nproc), dtype=np.float32),
                'obs_next': np.zeros((step_per_task, self.nproc, self.state_dim), dtype=np.float32)
            }
            return nproc_track

        sub_task_list = CircularList(sub_task_list)
        nproc_track = track_reset(self.step_per_task)
        env_idx_param = [sub_task_list.pop() for _ in range(self.nproc)]
        obs = self.env.reset()
        tstep = 0
        while True:
            actions, log_probs, mu, sigma = self.get_action(obs, env_idx_param)
            mapped_actions = self.map_action(actions)
            obs_next, rewards, dones, infos = self.env.step(mapped_actions)
            # 根据weight，自己计算reward和cost是多少
            rewards, costs = np.zeros(self.nproc), np.zeros(self.nproc)
            for idx, info in enumerate(infos):
                param = env_idx_param[idx]
                reward = info['reward_distance']*param[0] + info['reward_goal']*param[1]
                cost = info['cost_buttons']*param[2] + info['cost_gremlins']*param[3] + info['cost_hazards']*param[4]
                rewards[idx] = reward
                costs[idx] = cost

            nproc_track['obs'][tstep] = obs
            nproc_track['act'][tstep] = actions
            nproc_track['mu'][tstep] = mu
            nproc_track['sigma'][tstep] = sigma
            nproc_track['log_prob'][tstep] = log_probs
            nproc_track['rew'][tstep] = rewards
            nproc_track['cost'][tstep] = costs
            nproc_track['done'][tstep] = dones
            nproc_track['obs_next'][tstep] = obs_next
            tstep += 1
            sub_task_list.record(env_idx_param)

            terminal = any(dones) or (tstep >= self.step_per_task)
            if terminal:
                # 如果够了10000步，则每个赛道都要换下一组任务了
                if tstep >= self.step_per_task:
                    tstep = 0
                    # 把现有积累的数据给buffer，然后赛道重启
                    for k, data in nproc_track.items():
                        for idx in range(self.nproc):
                            param = env_idx_param[idx]
                            buffer[tuple(param)][k].append(data[:, idx])
                    nproc_track = track_reset(self.step_per_task)
                    # 如果所有人都够了10000步，就break
                    if sub_task_list.is_finish(self.step_per_task):
                        break
                    # 如果还有人没够，就循环访问下一组任务
                    env_idx_param = [sub_task_list.pop() for _ in range(self.nproc)]
                    obs_next = self.env.reset()
                # 如果还不到10000步但是有人done了，就对它reset继续采样
                else:
                    env_done_idx = np.where(dones)[0]
                    obs_reset = self.env.reset(env_done_idx)
                    obs_next = obs_next.copy()
                    obs_next[env_done_idx] = obs_reset
            obs = obs_next

        for param, data in buffer.items():
            data['avg_cumu_rew'] = np.sum(data['rew']) / (self.episode_per_proc*len(data['rew']))
            data['avg_cumu_cost'] = np.sum(data['cost']) / (self.episode_per_proc*len(data['rew']))
            for k, trans in data.items():
                if k in ['avg_cumu_rew', 'avg_cumu_cost']:
                    continue
                data[k] = np.concatenate(trans, axis=0)
            
        return buffer

    def compute_gae(self, buffer):
        """
        数据预处理，计算vs, cost_vs，adv, cost_adv，target, cost_target，不带梯度
        """
        for task, data in buffer.items():
            with th.no_grad():
                vs, cost_vs, _, _, _ = self.evaluate(data['obs'], np.array(task))
                vs_next, cost_vs_next, _, _, _ = self.evaluate(data['obs_next'], np.array(task))
            vs_numpy, vs_next_numpy = vs.cpu().numpy(), vs_next.cpu().numpy()
            cost_vs_numpy, cost_vs_next_numpy = cost_vs.cpu().numpy(), cost_vs_next.cpu().numpy()
            adv_numpy = _gae_return(
                vs_numpy, vs_next_numpy, data['rew'], data['done'], self.gamma, self.lam
            )
            cost_adv_numpy = _gae_return(
                cost_vs_numpy, cost_vs_next_numpy, data['cost'], data['done'], self.cost_gamma, self.cost_lam
            )

            target = _discount_cumsum(data['rew'], data['done'], self.gamma)
            cost_target = _discount_cumsum(data['cost'], data['done'], self.cost_gamma)
            # target = adv_numpy + vs_numpy
            # cost_target = cost_adv_numpy + cost_vs_numpy
            data['vs'] = vs_numpy
            data['cost_vs'] = cost_vs_numpy
            data['adv'] = adv_numpy
            data['cost_adv'] = cost_adv_numpy
            data['target'] = target
            data['cost_target'] = cost_target
        return buffer

    def get_action(self, obs, params):
        """
        obs: n_env x 60
        params: n_env x 6
        """
        params = th.tensor(self.norm_params(np.array(params)), dtype=th.float32, device=self.device)
        obs = th.tensor(obs, dtype=th.float32, device=self.device)
        with th.no_grad():
            mu, sigma = self.actor(obs, params)
            dist = self.dist_fn(mu, sigma)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.cpu().numpy(), log_prob.cpu().numpy(), mu.cpu().numpy(), sigma.cpu().numpy()

    def norm_params(self, params):
        low, high = self.task_sche.low, self.task_sche.high
        return (params - low) / (high - low)

    def map_action(self, act):
        act = np.clip(act, -1.0, 1.0)
        return act

    def evaluate(self, batch_obs, param, batch_acts=None):
        batch_param = param.reshape(1, -1).repeat(batch_obs.shape[0], axis=0)
        batch_param = th.tensor(self.norm_params(batch_param), dtype=th.float32, device=self.device)
        batch_obs = th.tensor(batch_obs, dtype=th.float32, device=self.device)
        vs = self.critic(batch_obs, batch_param).squeeze()
        cost_vs = self.cost_critic(batch_obs, batch_param).squeeze()
        log_probs, mu, sigma = None, None, None
        if batch_acts is not None:
            if isinstance(batch_acts, np.ndarray):
                batch_acts = th.tensor(batch_acts, dtype=th.float32, device=self.device)
            mu, sigma = self.actor(batch_obs, batch_param)
            dist = self.dist_fn(mu, sigma)
            log_probs = dist.log_prob(batch_acts)
            mu = mu.detach().cpu().numpy()
            sigma = sigma.detach().cpu().numpy()
        return vs, cost_vs, log_probs, mu, sigma
