import numpy as np


class TaskScheduler:
    """
    1. 均匀sample多个weight构成task，wr采样4个，wc采样15个，l采样5个；组合成共300个task，保存
    2. 每次选出30个距离最近的task，计算平均梯度更新策略
    """
    def __init__(self, param_start=[0], param_end=[10], dist_type='gaussian'):
        self.start = np.array(param_start)
        self.end = np.array(param_end)
        self.mu = (self.start + self.end) / 2
        self.sigma = (self.mu - self.start) / 3
        self.cov = np.diag(self.sigma)**2
        self.dist_type = dist_type
        if dist_type == 'gaussian':
            self.param_dist = multivariate_normal(mean=self.mu, cov=self.cov)
        elif dist_type == 'uniform':
            self.param_dist = uniform(loc=self.start, scale=self.end-self.start)
        else:
            raise NotImplementedError

    def set_division(self, block_num):
        traj_params, param_percents = [], {}
        edge_num = np.sqrt(block_num)
        block_size = (self.end - self.start) / edge_num
        block_size_density = block_size.copy()
        block_size_friction = block_size.copy()
        block_size_friction[0] = 0
        block_size_density[1] = 0
        left = np.array(self.start)
        right = left + block_size_friction + block_size_density
        param = np.random.uniform(left, right)
        percent = self.integral(left, right)
        for N in range(block_num):
            traj_params.append(list(param))
            param_percents[tuple(param)] = percent
            if N % edge_num != edge_num - 1:
                left = left + block_size_friction
                right = right + block_size_friction
            else:
                left = left + block_size_density
                left[1] = self.start[1]
                right = left + block_size_friction + block_size_density
            param = np.random.uniform(left, right)
            percent = self.integral(left, right)
        return traj_params, param_percents

    def sample(self, size=(1, 2)):
        # size = num x k
        tmp = self.param_dist.rvs(size=size)
        min_param = self.start.reshape(1, -1).repeat(size[0], axis=0)
        max_param = self.end.reshape(1, -1).repeat(size[0], axis=0)
        return np.clip(tmp, min_param, max_param)


class CircularList():
    def __init__(self, params=[]):
        assert len(params), "empty list."
        self.params = params
        self.idx = 0
        self.param2idx = {tuple(param): idx for idx, param in enumerate(params)}
        self.param_flag = np.zeros(len(self.params))
    
    def pop(self):
        param = self.params[self.idx].copy()
        self.idx = (self.idx + 1) % len(self.params)
        return param
    
    def record(self, params=[]):
        assert params, "empty list."
        finish_param_idx = [self.param2idx[tuple(param)] for param in params]
        self.param_flag[finish_param_idx] += 1

    def is_finish(self, threshold=1):
        return all(self.param_flag >= threshold)
