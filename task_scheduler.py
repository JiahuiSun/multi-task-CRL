import numpy as np
import itertools


class TaskScheduler:
    """
    输出：[[1, 1, 1, 1, 1, 25], [2, 0, 1, 2, 0, 30]]
    """
    def __init__(self):
        self.wr_list = [[0.2, 1.8], [0.5, 1.5], [1, 1], [1.5, 0.5], [1.8, 0.2]]
        self.wc_list = [[0.5, 1, 1.5], [0.5, 1.5, 1], [1, 0.5, 1.5], [1, 1.5, 0.5], [1.5, 0.5, 1], [1.5, 1, 0.5], [1, 1, 1]]
        self.threshold_list = [15, 20, 25, 30]
        self.task_list = []
        for task in list(itertools.product(self.wr_list, self.wc_list, self.threshold_list)):
            self.task_list.append(task[0]+task[1]+[task[2]])
        self.low = np.array([[0, 0, 0, 0, 0, 15]])
        self.high = np.array([[2, 2, 3, 3, 3, 30]])
        self.task_list_norm = (np.array(self.task_list) - self.low) / (self.high - self.low)
    
    def random_subset(self, N=10):
        task_ids = np.random.choice(len(self.task_list), N, replace=False)
        return [self.task_list[i] for i in task_ids]

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
