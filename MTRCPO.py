import time
import pickle
import os
import torch as th
from torch import nn
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
        dist,
        n_epoch=1000,
        episode_per_task=10,
        value_clip=True,
        clip=0.2,
        repeat_per_collect=10,
        lr_actor=3e-4,
        lr_critic=1e-3,
        lr_penalty=5e-2,
        gamma=0.99,
        lam=0.97,
        cost_gamma=0.99, 
        cost_lam=0.97,
        norm_adv=True,
        recompute_adv=False,
        max_grad_norm=0.5,
        save_freq=10,
        log_freq=1,
        writer=None,
        device='cpu'
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
        self.dist = dist

        self.n_epoch = n_epoch
        self.episode_per_task = episode_per_task
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
        self.log_freq = log_freq
        self.writer = writer
        self.device = device

        self.step_per_episode = 1000
 
    def learn(self):
        st_time = time.time()
        writer = self.writer
        for epoch in range(self.n_epoch):
            st1 = time.time()
            # TODO: task scheduler，负责每次哪些task一起训练，多少个epoch后切换一波task
            sub_task_list = [[1, 1, 1, 1, 1, 25]]

            # 并行sample N x episode_per_task条trajectory，每条trajectory有1000个step
            buffer = self.rollout(sub_task_list=sub_task_list)

            # training
            buffer = self.compute_gae(buffer)
            for repeat in range(self.repeat_per_collect):
                if self.recompute_adv and repeat > 0:
                    buffer = self.compute_gae(buffer)

                total_actor_loss, total_critic_loss, total_cost_critic_loss = 0, 0, 0
                # 对每个task计算actor loss，然后取平均
                # 对每个task计算critic loss，然后取平均
                # 对每个task的累积cost取平均，然后计算penalty loss
                for param, data in buffer.items():
                    # Calculate loss for critic
                    value, cost_value, curr_log_probs = self.evaluate(data['obs'], np.array(param), data['act'])
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

                    # Calculate loss for actor
                    adv = th.tensor(data['adv'], dtype=th.float32, device=self.device)
                    cadv = th.tensor(data['cadv'], dtype=th.float32, device=self.device)
                    # NOTE: 不管是否recompute adv，old_log_probs都不变，都是原来执行时产生的值
                    old_log_probs = th.tensor(data['log_prob'], dtype=th.float32, device=self.device)
                    if self.norm_adv:
                        adv = (adv - adv.mean()) / adv.std()
                    ratios = th.exp(curr_log_probs - old_log_probs)
                    surr1 = ratios * adv
                    surr2 = th.clamp(ratios, 1 - self.clip, 1 + self.clip) * adv
                    surr_adv = -th.min(surr1, surr2).mean()
                    surr_cost = -(ratios * cadv).mean()
                    # TODO: 这里要不要对penalty隔离梯度？
                    actor_loss = surr_adv - penalty * surr_cost
                    
                    total_actor_loss += 1 / num_task * actor_loss
                    total_critic_loss += 1 / num_task * critic_loss
                    

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
                        self.critic.parameters(), 
                        max_norm=self.max_grad_norm
                    )
                self.critic_optim.step()

            end_time = time.time()
            # log everything
            writer.add_scalar('metric/avg_return', np.mean(traj_return), T)
            writer.add_scalar('metric/wst10_return', np.mean(traj_return[:len(traj_return)//10]), T)
            writer.add_scalar('metric/avg_length', np.mean(traj_length), T)
            writer.add_scalar('metric/E_bar', np.sum(traj_return*traj_weight), T)
            writer.add_scalar('loss/ev', np.mean(ev_list), T)
            writer.add_scalar('loss/actor_loss', np.mean(actor_loss_list), T)
            writer.add_scalar('loss/critic_loss', np.mean(critic_loss_list), T)
            if (T+1) % self.log_freq == 0:
                logger.logkv('time_rollout', st2-st1)
                logger.logkv('time_training', end_time-st2)
                logger.logkv('time_one_epoch', end_time-st1)
                logger.logkv('time_elapsed', time.time()-st_time)
                logger.logkv('epoch', T)
                logger.logkv('avg_return', np.mean(traj_return))
                logger.logkv('wst10_return', np.mean(traj_return[:len(traj_return)//10]))
                logger.logkv('avg_length', np.mean(traj_length))
                logger.logkv('E_bar', np.sum(traj_return*traj_weight))
                logger.logkv('ev', np.mean(ev_list))
                logger.logkv('actor_loss', np.mean(actor_loss_list))
                logger.logkv('critic_loss', np.mean(critic_loss_list))
                logger.dumpkvs()
            if (T+1) % self.save_freq == 0:
                actor_path = os.path.join(logger.get_dir(), f'actor-{T}.pth')
                critic_path = os.path.join(logger.get_dir(), f'critic-{T}.pth')
                th.save(self.actor.state_dict(), actor_path)
                th.save(self.critic.state_dict(), critic_path)
                traj_info_path = os.path.join(logger.get_dir(), f'seq_dict_list-{T}.pkl')
                with open(traj_info_path, 'wb') as fout:
                    pickle.dump(seq_dict_list, fout)
                # 保存obs的normalization参数
                obs_norms = {
                    'clipob': self.env.clipob,
                    'mean': self.env.ob_rms.mean,
                    'var': self.env.ob_rms.var+self.env.epsilon
                }
                norm_param_path = os.path.join(logger.get_dir(), f'norm_param-{T}.pkl')
                with open(norm_param_path, 'wb') as f:
                    pickle.dump(obs_norms, f)
        self.env.close()

    def rollout(self, sub_task_list=[]):
        """
        并行sample N x episode_per_task条trajectory，每条trajectory有1000个step；
        统计平均累积reward和平均累积cost
        """
        buffer = {
            tuple(param): {
                'obs': np.zeros((self.step_per_episode, self.state_dim)), 
                'act': np.zeros((self.step_per_episode, self.act_dim)), 
                'log_prob': np.zeros(self.step_per_episode), 
                'rew': np.zeros(self.step_per_episode), 
                'cost': np.zeros(self.step_per_episode),
                'done': np.zeros(self.step_per_episode),
                'obs_next': np.zeros((self.step_per_episode, self.state_dim))
            } 
            for param in sub_task_list
        }
        traj_params = CircularList(traj_params)

        # 多进程并行采样，直到所有参数都被采样过
        env_idx_param = {idx: traj_params.pop() for idx in range(self.n_cpu)}
        self.env.set_params(env_idx_param)
        obs = self.env.reset()
        while True:
            params = np.array(self.env.get_params())
            actions, log_probs = self.get_action(obs, params)
            mapped_actions = self.map_action(actions)
            obs_next, rewards, dones, infos = self.env.step(mapped_actions)
            
            for idx, param in env_idx_param.items():
                buffer[tuple(param)]['obs'].append(obs[idx])
                buffer[tuple(param)]['act'].append(actions[idx])
                buffer[tuple(param)]['log_prob'].append(log_probs[idx])
                buffer[tuple(param)]['rew'].append(rewards[idx])  # 这里的reward已经被归一化了
                buffer[tuple(param)]['done'].append(dones[idx])
                buffer[tuple(param)]['obs_next'].append(obs_next[idx].copy())
                buffer[tuple(param)]['real_rew'].append(infos[idx])

            if any(dones):
                env_done_idx = np.where(dones)[0]
                # 采样停止条件：每个param都采样完一条trajectory，可以修改
                traj_params.record([env_idx_param[idx] for idx in env_done_idx])
                if traj_params.is_finish(threshold=self.traj_per_param):
                    break
                env_new_param = {idx: traj_params.pop() for idx in env_done_idx}
                self.env.set_params(env_new_param)
                obs_reset = self.env.reset(env_done_idx)
                obs_next[env_done_idx] = obs_reset
                env_idx_param.update(env_new_param)
            obs = obs_next

        # 数据预处理
        for param, data in buffer.items():
            data['obs'] = np.array(data['obs'])
            data['act'] = np.array(data['act'])
            data['log_prob'] = np.array(data['log_prob'])
            data['rew'] = np.array(data['rew'])
            data['done'] = np.array(data['done'])
            data['obs_next'] = np.array(data['obs_next'])
            data['real_rew'] = np.array(data['real_rew'])
            done_idx = np.where(data['done'])[0]
            data['return'] = np.sum(data['real_rew'][:max(done_idx)+1]) / len(done_idx)
            data['length'] = (max(done_idx)+1) / len(done_idx)
        return buffer

    def compute_gae(self, buffer):
        """
        数据预处理，计算vs, cost_vs，adv, cost_adv，target, cost_target，不带梯度
        """
        for task, data in buffer.items():
            # compute GAE
            with th.no_grad():
                vs, _ = self.evaluate(data['obs'], np.array(task))
                vs_next, _ = self.evaluate(data['obs_next'], np.array(task))
            vs_numpy, vs_next_numpy = vs.cpu().numpy(), vs_next.cpu().numpy()
            adv_numpy = _gae_return(
                vs_numpy, vs_next_numpy, data['rew'], data['done'], self.gamma, self.gae_lambda
            )
            returns_numpy = adv_numpy + vs_numpy
            data['vs'] = vs_numpy
            data['adv'] = adv_numpy
            data['target'] = returns_numpy

        return buffer

    def get_action(self, obs, params):
        if self.add_param:
            params = self.norm_params(params)
            obs_params = np.concatenate((obs, params), axis=1)
        else:
            obs_params = obs
        obs_params = th.tensor(obs_params, dtype=th.float32, device=self.device)
        with th.no_grad():
            mu, sigma = self.actor(obs_params)
            dist = self.dist_fn(mu, sigma)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.cpu().numpy(), log_prob.cpu().numpy()

    def norm_params(self, params):
        # input: Nx2, output: Nx2
        mu, sigma = self.env_para_dist.mu, self.env_para_dist.sigma
        return (params - mu) / sigma

    def map_action(self, act):
        act = np.clip(act, -1.0, 1.0)
        if self.action_scaling:
            assert np.min(act) >= -1.0 and np.max(act) <= 1.0, \
                "action scaling only accepts raw action range = [-1, 1]"
            low, high = self.action_space.low, self.action_space.high
            act = low + (high - low) * (act + 1.0) / 2.0  # type: ignore
        return act

    def evaluate(self, batch_obs, param, batch_acts=None):
        if self.add_param:
            batch_param = param.reshape(-1, 2).repeat(batch_obs.shape[0], axis=0)
            batch_param = self.norm_params(batch_param)
            batch_obs_params = np.concatenate((batch_obs, batch_param), axis=1)
        else:
            batch_obs_params = batch_obs
        batch_obs_params = th.tensor(batch_obs_params, dtype=th.float32, device=self.device)
        vs = self.critic(batch_obs_params).squeeze()
        log_probs = None
        if batch_acts is not None:
            if isinstance(batch_acts, np.ndarray):
                batch_acts = th.tensor(batch_acts, dtype=th.float32, device=self.device)
            mu, sigma = self.actor(batch_obs_params)
            dist = self.dist_fn(mu, sigma)
            log_probs = dist.log_prob(batch_acts)
        return vs, log_probs
