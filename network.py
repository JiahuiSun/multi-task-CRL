import torch as th
from torch import nn
import torch.nn.functional as F
import numpy as np


class BaseNet(nn.Module):
    def __init__(
        self,
        state_shape,
        hidden_dim=256
    ) -> None:
        super().__init__()
        input_dim = int(np.prod(state_shape))
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        self.output_dim = hidden_dim
    
    def forward(self, obs):
        logits = self.fc(obs)
        return logits


class Actor(nn.Module):
    def __init__(
        self,
        base_net,
        action_shape
    ) -> None:
        super().__init__()
        self.base_net = base_net
        input_dim = getattr(base_net, "output_dim")
        self.output_dim = int(np.prod(action_shape))
        self.mu = nn.Linear(input_dim, self.output_dim)
        self.sigma_param = nn.Parameter(th.zeros(self.output_dim, 1))
    
    def forward(self, obs):
        logits = self.base_net(obs)
        mu = self.mu(logits)
        shape = [1] * len(mu.shape)
        shape[1] = -1
        sigma = (self.sigma_param.view(shape) + th.zeros_like(mu)).exp()
        return mu, sigma


class Critic(nn.Module):
    def __init__(
        self,
        base_net
    ) -> None:
        super().__init__()
        self.base_net = base_net
        input_dim = getattr(base_net, "output_dim")
        self.last = nn.Linear(input_dim, 1)

    def forward(self, obs):
        logits = self.base_net(obs)
        logits = self.last(logits)
        return logits


class Penalty(nn.Module):
    def __init__(self, penalty_init=1) -> None:
        super().__init__()
        penalty_init = np.log(max(np.exp(penalty_init)-1, 1e-8))
        self.penalty = nn.Parameter(th.tensor(penalty_init, dtype=th.float32))
    
    def forward(self):
        return F.softplus(self.penalty)