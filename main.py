import numpy as np
import argparse
import time
import os
import gym
import safety_gym
import torch as th
from torch import nn
from torch.distributions import Independent, Normal
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tianshou.env import DummyVectorEnv, SubprocVectorEnv

from network import BaseNet, Actor, Critic
from MTRCPO import MTRCPO
from task_scheduler import TaskScheduler


def main(args):
    # logger
    t0 = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = f'seed_{args.seed}_{t0}'
    log_dir = os.path.join(args.log_dir, args.task, 'RCPO-threshold-as-input', log_file)
    writer = SummaryWriter(log_dir)
    writer.add_text('args', f"{args}")

    # env
    env = gym.make(args.task)
    state_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    # train_envs = DummyVectorEnv(
    #     [lambda: gym.make(args.task) for _ in range(args.nproc)], norm_obs=args.norm_obs
    # )
    train_envs = SubprocVectorEnv(
        [lambda: gym.make(args.task) for _ in range(args.nproc)], norm_obs=args.norm_obs
    )
    task_sche = TaskScheduler(epoch_per_threshold=args.epoch_per_task)

    # seed
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    train_envs.seed(args.seed)

    # actor, critic, cost_critic, penalty
    base_a = BaseNet(state_shape, args.taskid_dim).to(args.device)
    base_r = BaseNet(state_shape, args.taskid_dim).to(args.device)
    base_c = BaseNet(state_shape, args.taskid_dim).to(args.device)
    actor = Actor(base_a, action_shape).to(args.device)
    critic = Critic(base_r).to(args.device)
    cost_critic = Critic(base_c).to(args.device)
    penalty_init = np.log(max(np.exp(args.penalty_init)-1, 1e-8))
    penalty = nn.Parameter(th.tensor(penalty_init, dtype=th.float32).to(args.device))
 
    def dist(*logits):
        return Independent(Normal(*logits), 1)

    # Parameter initialization for actor, critic, cost_critic
    th.nn.init.constant_(actor.sigma_param, -0.5)
    if args.param_init:
        for m in list(actor.modules()) + list(critic.modules()) + list(cost_critic.modules()):
            if isinstance(m, th.nn.Linear):
                # orthogonal initialization
                th.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                th.nn.init.zeros_(m.bias)
        # do last policy layer scaling, this will make initial actions have (close to)
        # 0 mean and std, and will help boost performances,
        # see https://arxiv.org/abs/2006.05990, Fig.24 for details
        for m in actor.mu.modules():
            if isinstance(m, th.nn.Linear):
                th.nn.init.zeros_(m.bias)
                m.weight.data.copy_(0.01 * m.weight.data)

    # optimizers for actor, critic, cost_critic
    actor_optim = Adam(actor.parameters(), lr=args.lr_actor)
    critic_optim = Adam(list(critic.parameters())+list(cost_critic.parameters()), lr=args.lr_critic)
    penalty_optim = Adam([penalty], lr=args.lr_penalty)

    agent = MTRCPO(
        env=train_envs,
        task_sche=task_sche,
        state_shape=state_shape,
        action_shape=action_shape,
        actor=actor,
        critic=critic,
        cost_critic=cost_critic,
        penalty=penalty,
        actor_optim=actor_optim,
        critic_optim=critic_optim,
        penalty_optim=penalty_optim,
        dist_fn=dist,
        writer=writer,
        device=args.device,
        # 参数设置参考TensorFlow
        n_epoch=args.n_epoch,
        epoch_per_task=args.epoch_per_task,
        episode_per_proc=args.episode_per_proc,
        repeat_per_collect=args.repeat_per_collect,
        kl_stop=args.kl_stop,
        norm_adv=args.norm_adv,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        lr_penalty=args.lr_penalty
    )
    agent.learn()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Multi-task Constrained RL")
    parser.add_argument('--task', type=str, default='Safexp-PointGoal1-v0')
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--nproc', type=int, default=10)
    parser.add_argument('--log_dir', type=str, default='output')
    parser.add_argument(
        '--device', type=str, default='cuda' if th.cuda.is_available() else 'cpu'
    )
    parser.add_argument('--penalty_init', type=float, default=1)
    parser.add_argument('--n_encoder', type=int, default=4)
    parser.add_argument('--taskid_dim', type=int, default=10)
    parser.add_argument('--norm_obs', action='store_true')
    parser.add_argument('--norm_adv', action='store_false')
    parser.add_argument('--kl_stop', action='store_false')
    parser.add_argument('--param_init', action='store_true')
    parser.add_argument('--n_epoch', type=int, default=2000)
    parser.add_argument('--episode_per_proc', type=int, default=2)
    parser.add_argument('--epoch_per_task', type=int, default=100, help='#epoch before switch to the next task')
    parser.add_argument('--repeat_per_collect', type=int, default=20)
    parser.add_argument('--lr_actor', type=float, default=3e-4)
    parser.add_argument('--lr_critic', type=float, default=1e-3)
    parser.add_argument('--lr_penalty', type=float, default=5e-2)
    args = parser.parse_args()

    main(args)
