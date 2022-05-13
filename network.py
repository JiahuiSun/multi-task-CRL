import torch as th
from torch import nn
import torch.nn.functional as F
import numpy as np


class Linear(nn.Module):
    def __init__(
        self, num_experts: int, in_features: int, out_features: int, bias: bool = True
    ):
        """torch.nn.Linear layer extended for use as a mixture of experts.

        Args:
            num_experts (int): number of experts in the mixture.
            in_features (int): size of each input sample for one expert.
            out_features (int): size of each output sample for one expert.
            bias (bool, optional): if set to ``False``, the layer will
                not learn an additive bias. Defaults to True.
        """
        super().__init__()
        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            th.rand(self.num_experts, self.in_features, self.out_features)
        )
        if bias:
            self.bias = nn.Parameter(th.rand(self.num_experts, 1, self.out_features))
            self.use_bias = True
        else:
            self.use_bias = False

    def forward(self, x):
        if self.use_bias:
            return x.matmul(self.weight) + self.bias
        else:
            return x.matmul(self.weight)

    def extra_repr(self):
        return f"num_experts={self.num_experts}, in_features={self.in_features}, out_features={self.out_features}, bias={self.use_bias}"


class SoftModularizedMLP(nn.Module):
    def __init__(
        self,
        num_experts: int,
        obs_dim: int,
        task_dim: int,
        out_features: int,
        num_layers: int,
        hidden_features: int,
        bias: bool = True,
    ):
        """Class to implement the actor/critic in
        'Multi-Task Reinforcement Learning with Soft Modularization' paper.
        It is similar to layers.FeedForward but allows selection of expert
        at each layer.
        """
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_features),
            nn.Tanh(),
            nn.Linear(hidden_features, hidden_features)
        )
        self.task_encoder = nn.Sequential(
            nn.Linear(task_dim, hidden_features)
        )

        layers = []
        current_in_features = hidden_features
        self.output_dim = out_features
        for _ in range(num_layers - 1):
            linear = Linear(
                num_experts=num_experts,
                in_features=current_in_features,
                out_features=hidden_features,
                bias=bias,
            )
            layers.append(nn.Sequential(linear, nn.Tanh()))
            # Each layer is a combination of a moe layer and ReLU.
            current_in_features = hidden_features
        linear = Linear(
            num_experts=num_experts,
            in_features=current_in_features,
            out_features=out_features,
            bias=bias,
        )
        layers.append(linear)
        self.layers = nn.ModuleList(layers)
        self.routing_network = RoutingNetwork(
            in_features=hidden_features,
            hidden_features=hidden_features,
            num_layers=num_layers - 1,
            num_experts_per_layer=num_experts,
        )

    def forward(self, obs, taskid):
        obs = self.obs_encoder(obs)
        taskid = self.task_encoder(taskid)
        probs = self.routing_network(obs, taskid)
        # (num_layers, batch, num_experts, num_experts)
        probs = probs.permute(0, 2, 3, 1)
        # (num_layers, num_experts, num_experts, batch)
        num_experts = probs.shape[1]
        inp = obs
        # (batch, dim1)
        for index, layer in enumerate(self.layers[:-1]):
            p = probs[index]
            # (num_experts, num_experts, batch)
            inp = layer(inp)
            # (num_experts, batch, dim2)
            _out = p.unsqueeze(-1) * inp.unsqueeze(0).repeat(num_experts, 1, 1, 1)
            # (num_experts, num_experts, batch, dim2)
            inp = _out.sum(dim=1)
            # (num_experts, batch, dim2)
        out = self.layers[-1](inp).mean(dim=0)
        # (batch, out_dim)
        return out


class RoutingNetwork(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        num_experts_per_layer: int,
        num_layers: int,
    ) -> None:
        """Class to implement the routing network in
        'Multi-Task Reinforcement Learning with Soft Modularization' paper.
        """
        super().__init__()

        self.num_experts_per_layer = num_experts_per_layer

        self.projection_before_routing = nn.Linear(
            in_features=in_features,
            out_features=hidden_features,
        )

        self.W_d = nn.ModuleList(
            [
                nn.Linear(
                    in_features=hidden_features,
                    out_features=self.num_experts_per_layer ** 2,
                )
                for _ in range(num_layers)
            ]
        )

        self.W_u = nn.ModuleList(
            [
                nn.Linear(
                    in_features=self.num_experts_per_layer ** 2,
                    out_features=hidden_features,
                )
                for _ in range(num_layers - 1)
            ]
        )  # the first layer does not need W_u

        self._softmax = nn.Softmax(dim=2)

    def _process_logprob(self, logprob):
        logprob_shape = logprob.shape
        logprob = logprob.reshape(
            logprob_shape[0], self.num_experts_per_layer, self.num_experts_per_layer
        )
        # logprob[:][i][j] == weight (probability )of the ith module (in current layer)
        # for contributing to the jth module in the next layer.
        # Since the ith module has to divide its weight among all modules in the
        # next layer, logprob[batch_index][i][:] to 1
        prob = self._softmax(logprob)
        return prob

    def forward(self, obs, task_info):
        inp = self.projection_before_routing(obs * task_info)
        p = self.W_d[0](F.tanh(inp))  # batch x num_experts ** 2
        prob = [p]
        for W_u, W_d in zip(self.W_u, self.W_d[1:]):
            p = W_d(F.tanh((W_u(prob[-1]) * inp)))
            prob.append(p)
        prob_tensor = th.cat(
            [self._process_logprob(logprob=logprob).unsqueeze(0) for logprob in prob],
            dim=0,
        )
        return prob_tensor


class BaseNet(nn.Module):
    def __init__(
        self,
        state_shape,
        taskid_dim
    ) -> None:
        super().__init__()
        input_dim = int(np.prod(state_shape)) + taskid_dim
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh()
        )
        self.output_dim = 256
    
    def forward(self, obs, taskid):
        obs_task = th.cat([obs, taskid], dim=1)
        return self.fc(obs_task)


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