import torch as th
from torch import nn
import torch.nn.functional as F
import numpy as np


class BaseNet(nn.Module):
    def __init__(
        self,
        state_shape,
        taskid_dim,
        n_encoder=10
    ) -> None:
        super().__init__()
        input_dim = int(np.prod(state_shape))
        self.state_encoder_list = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.Tanh(),
                nn.Linear(256, 256)
            )
            for _ in range(n_encoder)
        ])
        self.task_encoder = nn.Sequential(
            nn.Linear(taskid_dim, 256)
        )
        self.mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.Tanh()
        )
        self.output_dim = 512
    
    def forward(self, obs, taskid):
        state_embs = th.stack([state_encoder(obs) for state_encoder in self.state_encoder_list], dim=0)  # KBF
        state_embs = state_embs.permute(1, 0, 2)  # BKF
        task_emb = self.task_encoder(taskid)  # BF
        task_emb_nograd = task_emb.detach()
        atten_logit = state_embs.matmul(task_emb_nograd.unsqueeze(-1))  # BK1
        atten_weight = F.softmax(atten_logit, dim=1)
        state_emb = atten_weight.permute(0, 2, 1).matmul(state_embs).squeeze(1)  # BF
        state_emb = self.mlp(state_emb)
        state_task_emb = th.cat([task_emb, state_emb], dim=1)  # B(2F)
        return state_task_emb


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
    
    def forward(self, obs, taskid):
        state_task_emb = self.base_net(obs, taskid)
        mu = self.mu(state_task_emb)
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

    def forward(self, obs, taskid):
        state_task_emb = self.base_net(obs, taskid)
        logits = self.last(state_task_emb)
        return logits


class Penalty(nn.Module):
    def __init__(self, penalty_init=1) -> None:
        super().__init__()
        penalty_init = np.log(max(np.exp(penalty_init)-1, 1e-8))
        self.penalty = nn.Parameter(th.tensor(penalty_init, dtype=th.float32))
    
    def forward(self):
        # penalty = nn.Parameter
        # optim = Adam(penalty)

        # loss = -penalty * ()
        # loss.backward()
        # optim.step(), get new penalty
        # penalty = nn.Parameter(F.relu(new penalty))
        # TODO: 我的疑惑是，梯度更新是对clip之前的penalty还是之后的penalty？
        # 如果按照TensorFlow是对clip之前的，而我自己的理解是对clip之后的
        return self.penalty, F.relu(self.penalty).detach()