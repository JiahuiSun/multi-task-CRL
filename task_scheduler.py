import numpy as np
import itertools


class TaskScheduler:
    """
    1. 均匀sample多个weight构成task，wr采样4个，wc采样15个，l采样5个；组合成共300个task，保存
    2. 每次选出30个距离最近的task，计算平均梯度更新策略
    3. 输出：[[1, 1, 1, 1, 1, 25], [2, 0, 1, 2, 0, 30]]
    TODO:
    先随便写几个task，然后先debug吧，确定逻辑没问题了再来写这个
    """
    def __init__(self):
        self.wr_list = [[0.5, 1.5], [1, 1], [1.5, 0.5]]
        self.wc_list = [[0, 1, 2], [1, 0, 2], [1, 2, 0]]
        self.threshold_list = [10, 20, 30, 40, 50]
        self.task_list = list(itertools.product(self.wr_list, self.wc_list, self.threshold_list))
        self.low = np.array([[0, 0, 0, 0, 0, 10]])
        self.high = np.array([[2, 2, 3, 3, 3, 50]])
    
    def subset(self, N=10):
        sub_task_list = []
        for i in range(N):
            task = self.task_list[i]
            sub_task_list.append((task[0]+task[2]).append(task[1]))
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
