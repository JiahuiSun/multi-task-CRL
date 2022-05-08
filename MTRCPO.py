import time
import pickle
import os
from tqdm import tqdm
import torch as th
import torch.nn.functional as F
from numba import njit
import numpy as np


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


class MTRCPO:
    def __init__(
        self,
        env,
        cost_lim,
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
        save_freq=10,
        writer=None,
        device='cpu',
        # 接下来的参数可调
        n_epoch=1000,
        episode_per_proc=10,
        value_clip=False,
        clip=0.2,
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
        self.cost_lim = cost_lim
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
        self.step_per_proc = self.step_per_episode * self.episode_per_proc
 
    def learn(self):
        st_time = time.time()
        all_epoch_cost = 0
        for epoch in tqdm(range(self.n_epoch)):
            st1 = time.time()
            buffer = self.rollout()
            st2 = time.time()

            # training
            buffer = self.compute_gae(buffer)
            penalty = F.softplus(self.penalty.detach())
            for repeat in range(self.repeat_per_collect):
                if self.recompute_adv and repeat > 0:
                    buffer = self.compute_gae(buffer)

                # Calculate loss for critic
                value, cost_value, curr_log_probs = self.evaluate(buffer['obs'], buffer['act'])
                target = th.tensor(buffer['target'], dtype=th.float32, device=self.device)
                cost_target = th.tensor(buffer['cost_target'], dtype=th.float32, device=self.device)
                # NOTE: 只有不recompute adv时，value clip才有意义
                if self.value_clip:
                    vs = th.tensor(buffer['vs'], dtype=th.float32, device=self.device)
                    v_clip = vs + (value - vs).clamp(-self.clip, self.clip)
                    vf1 = (target - value).pow(2)
                    vf2 = (target - v_clip).pow(2)
                    critic_loss = th.max(vf1, vf2).mean()

                    vs = th.tensor(buffer['cost_vs'], dtype=th.float32, device=self.device)
                    v_clip = vs + (cost_value - vs).clamp(-self.clip, self.clip)
                    vf1 = (cost_target - cost_value).pow(2)
                    vf2 = (cost_target - v_clip).pow(2)
                    cost_critic_loss = th.max(vf1, vf2).mean()
                else:
                    critic_loss = (target - value).pow(2).mean()
                    cost_critic_loss = (cost_target - cost_value).pow(2).mean()
                total_critic_loss = critic_loss + cost_critic_loss

                # Calculate loss for actor
                adv = th.tensor(buffer['adv'], dtype=th.float32, device=self.device)
                cost_adv = th.tensor(buffer['cost_adv'], dtype=th.float32, device=self.device)
                # NOTE: 不管是否recompute adv，old_log_probs都不变，都是原来执行时产生的值
                old_log_probs = th.tensor(buffer['log_prob'], dtype=th.float32, device=self.device)
                if self.norm_adv:
                    adv = (adv - adv.mean()) / adv.std()
                ratios = th.exp(curr_log_probs - old_log_probs)
                surr1 = ratios * adv
                surr2 = th.clamp(ratios, 1 - self.clip, 1 + self.clip) * adv
                surr_rew_loss = -th.min(surr1, surr2).mean()
                surr_cost_loss = -(ratios * cost_adv).mean()
                total_actor_loss = surr_rew_loss - penalty * surr_cost_loss
                # why?
                total_actor_loss /= (1 + penalty)
                
                self.actor_optim.zero_grad()
                total_actor_loss.backward()
                if self.max_grad_norm:  # clip large gradient
                    th.nn.utils.clip_grad_norm_(
                        self.actor.parameters(), 
                        max_norm=self.max_grad_norm
                    )
                self.actor_optim.step()

                self.critic_optim.zero_grad()
                total_critic_loss.backward()
                if self.max_grad_norm:  # clip large gradient
                    th.nn.utils.clip_grad_norm_(
                        list(self.critic.parameters())+list(self.cost_critic.parameters()), 
                        max_norm=self.max_grad_norm
                    )
                self.critic_optim.step()

            # update penalty
            penalty_loss = -self.penalty * (buffer['avg_cumu_cost'] - self.cost_lim)
            self.penalty_optim.zero_grad()
            penalty_loss.backward()
            self.penalty_optim.step()

            # log everything
            end_time = time.time()
            all_epoch_cost += buffer['avg_cumu_cost']
            cost_rate = all_epoch_cost / ((epoch+1)*self.step_per_episode)

            self.writer.add_scalar('param/threshold', self.cost_lim, epoch)
            self.writer.add_scalar('param/penalty', penalty, epoch)
            self.writer.add_scalar('metric/avg_cumu_rew', buffer['avg_cumu_rew'], epoch)
            self.writer.add_scalar('metric/avg_cumu_cost', buffer['avg_cumu_cost'], epoch)
            self.writer.add_scalar('metric/cost_rate', cost_rate, epoch)
            self.writer.add_scalar('loss/all_task_actor_loss', total_actor_loss.item(), epoch)
            self.writer.add_scalar('loss/total_critic_loss', total_critic_loss.item(), epoch)
            self.writer.add_scalar('time/rollout', st2-st1, epoch)
            self.writer.add_scalar('time/training', end_time-st2, epoch)
            self.writer.add_scalar('time/elapsed', time.time()-st_time, epoch)
            
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

    def rollout(self):
        """
        1. 共sample n_task x episode_per_task条episode，每条episode有1000个step；
            - 逻辑：nproc个task并行采样episode_per_task条episode，换下一批task
        2. 统计平均累积reward和平均累积cost
        """
        buffer = {
            'obs': np.zeros((self.step_per_proc, self.nproc, self.state_dim), dtype=np.float32),  # T x N x D
            'act': np.zeros((self.step_per_proc, self.nproc, self.act_dim), dtype=np.float32),
            'log_prob': np.zeros((self.step_per_proc, self.nproc), dtype=np.float32),
            'rew': np.zeros((self.step_per_proc, self.nproc), dtype=np.float32),
            'cost': np.zeros((self.step_per_proc, self.nproc), dtype=np.float32),
            'done': np.zeros((self.step_per_proc, self.nproc), dtype=np.float32),
            'obs_next': np.zeros((self.step_per_proc, self.nproc, self.state_dim), dtype=np.float32)
        }

        obs = self.env.reset()
        tstep = 0
        while True:
            actions, log_probs = self.get_action(obs)
            mapped_actions = self.map_action(actions)
            obs_next, rewards, dones, infos = self.env.step(mapped_actions)
            costs = np.array([info['cost'] for info in infos])

            buffer['obs'][tstep] = obs
            buffer['act'][tstep] = actions
            buffer['log_prob'][tstep] = log_probs
            buffer['rew'][tstep] = rewards
            buffer['cost'][tstep] = costs
            buffer['done'][tstep] = dones
            buffer['obs_next'][tstep] = obs_next
            tstep += 1

            terminal = any(dones) or (tstep >= self.step_per_proc)
            if terminal:
                # 如果够了10000步，则每个赛道都要换下一组任务了
                if tstep >= self.step_per_proc:
                    break
                # 如果还不到10000步但是有人done了，就对它reset继续采样
                else:
                    env_done_idx = np.where(dones)[0]
                    obs_reset = self.env.reset(env_done_idx)
                    obs_next = obs_next.copy()
                    obs_next[env_done_idx] = obs_reset
            obs = obs_next

        for key, data in buffer.items():
            buffer[key] = np.concatenate([data[:, i] for i in range(self.nproc)], axis=0)
        buffer['avg_cumu_rew'] = np.sum(buffer['rew']) / (self.episode_per_proc*self.nproc)
        buffer['avg_cumu_cost'] = np.sum(buffer['cost']) / (self.episode_per_proc*self.nproc)

        return buffer

    def compute_gae(self, buffer):
        """
        数据预处理，计算vs, cost_vs，adv, cost_adv，target, cost_target，不带梯度
        """
        with th.no_grad():
            vs, cost_vs, _ = self.evaluate(buffer['obs'])
            vs_next, cost_vs_next, _ = self.evaluate(buffer['obs_next'])
        vs_numpy, vs_next_numpy = vs.cpu().numpy(), vs_next.cpu().numpy()
        cost_vs_numpy, cost_vs_next_numpy = cost_vs.cpu().numpy(), cost_vs_next.cpu().numpy()
        adv_numpy = _gae_return(
            vs_numpy, vs_next_numpy, buffer['rew'], buffer['done'], self.gamma, self.lam
        )
        cost_adv_numpy = _gae_return(
            cost_vs_numpy, cost_vs_next_numpy, buffer['cost'], buffer['done'], self.cost_gamma, self.cost_lam
        )
        target = adv_numpy + vs_numpy
        cost_target = cost_adv_numpy + cost_vs_numpy
        buffer['vs'] = vs_numpy
        buffer['cost_vs'] = cost_vs_numpy
        buffer['adv'] = adv_numpy
        buffer['cost_adv'] = cost_adv_numpy
        buffer['target'] = target
        buffer['cost_target'] = cost_target
        return buffer

    def get_action(self, obs):
        """
        obs: n_env x 60
        params: n_env x 6
        """
        obs = th.tensor(obs, dtype=th.float32, device=self.device)
        with th.no_grad():
            mu, sigma = self.actor(obs)
            dist = self.dist_fn(mu, sigma)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.cpu().numpy(), log_prob.cpu().numpy()

    def map_action(self, act):
        act = np.clip(act, -1.0, 1.0)
        return act

    def evaluate(self, batch_obs, batch_acts=None):
        batch_obs = th.tensor(batch_obs, dtype=th.float32, device=self.device)
        vs = self.critic(batch_obs).squeeze()
        cost_vs = self.cost_critic(batch_obs).squeeze()
        log_probs = None
        if batch_acts is not None:
            if isinstance(batch_acts, np.ndarray):
                batch_acts = th.tensor(batch_acts, dtype=th.float32, device=self.device)
            mu, sigma = self.actor(batch_obs)
            dist = self.dist_fn(mu, sigma)
            log_probs = dist.log_prob(batch_acts)
        return vs, cost_vs, log_probs
