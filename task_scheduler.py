import numpy as np
import itertools


class TaskScheduler:
    """
    输出：[[1, 1, 1, 1, 1, 25], [2, 0, 1, 2, 0, 30]]
    """
    def __init__(self, epoch_per_threshold=100):
        self.epoch_per_threshold = epoch_per_threshold
        self.wr_list = [[1, 1]]
        self.wc_list = [[1, 1, 1], [0.5, 1, 1.5], [0.5, 1.5, 1], [1, 1.5, 0.5], [1, 0.5, 1.5], [1.5, 0.5, 1], [1.5, 1, 0.5]]
        self.threshold_list = [20, 30, 25, 15]
        self.task_list = []
        for task in list(itertools.product(self.wr_list, self.wc_list, self.threshold_list)):
            self.task_list.append(task[0]+task[1]+[task[2]])
        self.low = np.array([[0, 0, 0, 0, 0, 15]])
        self.high = np.array([[2, 2, 3, 3, 3, 30]])
        self.t_idx = 0
        self.task = self.task_list[self.t_idx]
    
    def update(self, epoch):
        if epoch % self.epoch_per_threshold == 0:
            self.task = self.task_list[self.t_idx%len(self.task_list)]
            self.t_idx += 1
        return self.task

    def subset(self, N=None):
        if N is None:
            return self.task_list
        else:
            sub_tasks = np.random.permutation(len(self.task_list))[:N]
            return [self.task_list[i] for i in sub_tasks]


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
        # 记录一次，说明这些参数的transition增加了一步
        assert params, "empty list."
        finish_param_idx = [self.param2idx[tuple(param)] for param in params]
        self.param_flag[finish_param_idx] += 1

    def is_finish(self, threshold=1):
        return all(self.param_flag >= threshold)
