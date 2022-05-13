import numpy as np
import argparse
import time
import gym
import safety_gym
import torch as th
from torch.distributions import Independent, Normal
from tianshou.env import DummyVectorEnv, SubprocVectorEnv

from network import BaseNet, Actor
from MTRCPO import MTRCPO
from task_scheduler import TaskScheduler


class Agent(MTRCPO):
    def __init__(self, env, state_shape,
        action_shape, actor, dist_fn, device='cpu',  episode_per_proc=10) -> None:
        self.env = env
        self.state_dim = np.prod(state_shape)
        self.act_dim = np.prod(action_shape)
        self.actor = actor
        self.dist_fn = dist_fn
        self.device = device
        self.episode_per_proc = episode_per_proc

        self.nproc = len(env)
        self.step_per_episode = 1000
        self.step_per_proc = self.step_per_episode * self.episode_per_proc


def main(args):
    # env
    env = gym.make(args.task)
    state_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    # test_envs = DummyVectorEnv(
    #     [lambda: gym.make(args.task) for _ in range(args.nproc)], norm_obs=args.norm_obs
    # )
    test_envs = SubprocVectorEnv(
        [lambda: gym.make(args.task) for _ in range(args.nproc)], norm_obs=args.norm_obs
    )

    # task preprocess
    task_sche = TaskScheduler()
    # 我告诉你我想测试哪个threshold，比如100，你给我返回对应的最近的one-hot
    task = task_sche.parse(args.threshold)

    # seed
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    test_envs.seed(args.seed)

    # actor
    base_a = BaseNet(state_shape, args.taskid_dim, n_encoder=args.n_encoder).to(args.device)
    actor = Actor(base_a, action_shape).to(args.device)
    actor.load_state_dict(th.load(f"{args.actor_path}", map_location=args.device))

    def dist(*logits):
        return Independent(Normal(*logits), 1)

    agent = Agent(
        env=test_envs,
        state_shape=state_shape,
        action_shape=action_shape,
        actor=actor,
        dist_fn=dist,
        device=args.device,
        episode_per_proc=args.episode_per_proc
    )
    st = time.time()
    buffer = agent.rollout(task)
    print(f"avg_cumu_rew: {buffer['avg_cumu_rew']}")
    print(f"avg_cumu_cost: {buffer['avg_cumu_cost']}")
    print(f"time: {time.time() - st:.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Multi-task Constrained RL")
    parser.add_argument('--task', type=str, default='Safexp-PointGoal1-v0')
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--nproc', type=int, default=10)
    parser.add_argument('--actor_path', type=str, default='output')
    parser.add_argument(
        '--device', type=str, default='cuda' if th.cuda.is_available() else 'cpu'
    )
    parser.add_argument('--taskid_dim', type=int, default=20)
    parser.add_argument('--n_encoder', type=int, default=4)
    parser.add_argument('--norm_obs', action='store_true')
    parser.add_argument('--episode_per_proc', type=int, default=10)
    parser.add_argument('--threshold', type=float, default=25)
    args = parser.parse_args()

    main(args)
