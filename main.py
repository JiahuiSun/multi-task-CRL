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
    log_dir = os.path.join(args.log_dir, args.task, 'MTRCPO', log_file)
    writer = SummaryWriter(log_dir)
    writer.add_text('args', f"{args}")

    # env
    env = gym.make(args.task)
    state_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    train_envs = DummyVectorEnv(
        [lambda: gym.make(args.task) for _ in range(args.training_num)], norm_obs=True
    )
    # train_envs = SubprocVectorEnv(
    #     [lambda: gym.make(args.task) for _ in range(args.training_num)], norm_obs=True
    # )
    # TODO: 千万记得保存norm_obs
    task_sche = TaskScheduler()

    # seed
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    train_envs.seed(args.seed)

    # actor, critic, cost_critic, penalty and their optimizers
    base_a = BaseNet(state_shape, args.taskid_dim, n_encoder=args.n_encoder)
    base_r = BaseNet(state_shape, args.taskid_dim, n_encoder=args.n_encoder)
    base_c = BaseNet(state_shape, args.taskid_dim, n_encoder=args.n_encoder)
    actor = Actor(base_a, action_shape).to(args.device)
    critic = Critic(base_r).to(args.device)
    cost_critic = Critic(base_c).to(args.device)
    penalty = nn.Sequential(
        nn.Parameter(np.log(max(np.exp(args.penalty_init)-1, 1e-8))),
        nn.Softplus()
    ).to(args.device)
 
    def dist(*logits):
        return Independent(Normal(*logits), 1)

    th.nn.init.constant_(actor.sigma_param, -0.5)
    for m in list(actor.modules()) + list(critic.modules()):
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

    actor_optim = Adam(actor.parameters(), lr=args.lr_actor)
    critic_optim = Adam(list(critic.parameters())+list(cost_critic.parameters()), lr=args.lr_critic)
    penalty_optim = Adam(penalty.parameters(), lr=args.lr_penalty)

    agent = MTRCPO(
        env=train_envs,
        task_sche=task_sche,
        actor=actor,
        critic=critic,
        cost_critic=cost_critic,
        penalty=penalty,
        actor_optim=actor_optim,
        critic_optim=critic_optim,
        penalty_optim=penalty_optim,
        dist=dist,
        writer=writer
    )
    agent.learn(args.n_epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Distribution Robust RL")
    parser.add_argument('--task', type=str, default='Safexp-PointGoal1-v0')
    parser.add_argument('--seed', type=int, default=12)
    parser.add_argument('--training_num', type=int, default=4)
    parser.add_argument('--log_dir', type=str, default='output')
    parser.add_argument(
        '--device', type=str, default='cuda' if th.cuda.is_available() else 'cpu'
    )

    parser.add_argument('--taskid_dim', type=int, default=6)
    parser.add_argument('--n_encoder', type=int, default=10)
    parser.add_argument('--n_epoch', type=int, default=10)
    parser.add_argument('--lr_actor', type=float, default=3e-4)
    parser.add_argument('--lr_critic', type=float, default=3e-4)
    parser.add_argument('--lr_penalty', type=float, default=3e-4)
    parser.add_argument('--penalty_init', type=float, default=1)
    

    parser.add_argument('--recompute_adv', type=int, default=0, help='whether to recompute adv')
    parser.add_argument('--value_clip', type=int, default=1, help='whether to clip value')
    parser.add_argument('--add_param', type=int, default=1, help='whether to add param')

    parser.add_argument('--max_grad_norm', type=float, default=0.5)
    parser.add_argument('--traj_per_param', type=float, default=1)
    parser.add_argument('--action_scaling', type=int, default=0, help='whether to scale action')
    parser.add_argument('--block_num', type=int, default=100)
    parser.add_argument('--param_dist', type=str, default='gaussian', choices=['gaussian', 'uniform'])
    parser.add_argument('--clip', type=float, default=0.2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--total_iters', type=int, default=1000)
    parser.add_argument('--norm_adv', type=int, default=1, help='whether to norm adv')
    parser.add_argument('--repeat_per_collect', type=float, default=10)
    parser.add_argument('--log_freq', type=int, default=1)
    parser.add_argument('--save_freq', type=int, default=100)
    args = parser.parse_args()

    main(args)
