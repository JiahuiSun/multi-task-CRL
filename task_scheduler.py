import numpy as np
import itertools


class TaskScheduler:
    """
    输出：[[1, 1, 1, 1, 1, 25], [2, 0, 1, 2, 0, 30]]
    """
    def __init__(self, epoch_per_threshold=100):
        self.epoch_per_threshold = epoch_per_threshold
        self.threshold_list = np.arange(0, 101, 3)
        int2binary = lambda x: [int(tmp) for tmp in format(x, 'b').zfill(6)]
        self.task_list = [int2binary(x) for x in self.threshold_list]
        self.binary2int = {tuple(int2binary(x)): x for x in self.threshold_list}
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
            sub_task_list = []
            for i in range(N):
                task = self.task_list[i]
                sub_task_list.append(task[0]+task[1]+[task[2]])
            return sub_task_list


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
