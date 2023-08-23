import datetime
import logging
import os
import random
from pathlib import Path

import torch.nn.functional as F
import numpy as np
import torch
from IPython import display
from matplotlib import pyplot as plt
from torch import nn


def sequence_mask(X, valid_len, value=0.0):
    """在序列中屏蔽不相关的项
    数据要求:
    X: (batch_size, seq_len, ****(任意,只确保前两个))
    valid_len: (batch_size,) 确保只能是单维度向量
    """
    maxlen = X.size(1)
    # 通过广播机制得到一个只含true/false的类同(X.shape(0),X.shape(1))的矩阵,第一个None广播至batch_size,第二个None广播至seq_size
    mask = torch.arange(maxlen, dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    # 通过广播机制将X的前两个维度进行遮盖,后面的维度视为一体
    X[~mask] = value
    return X


def masked_softmax(X, valid_lens):
    """通过在最后⼀个轴上掩蔽元素来执⾏softmax操作
    数据要求:
    X: (batch_size, some_batch_size, seq_len)
    valid_lens: (batch_size,) | (batch_size, some_batch_size)
    """
    # X:3D张量，valid_lens:1D或2D张量
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])  # 此时假定X是3D张量,也就是(batch1, some_batch2, seq_len)
        else:
            valid_lens = valid_lens.reshape(-1)
        # 最后⼀轴上被掩蔽的元素使⽤⼀个⾮常⼤的负值替换，从⽽其softmax输出为0
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)


def get_logger(log_path='./log/gxl.log'):
    """
    将日志输出到日志文件和控制台
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler(
        filename=log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # 创建一个handler，用于将日志输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)
    return logger


def set_random_seed(seed, cuda):
    """
    设置训练的随机种子
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def top_k_top_p_filtering(logits, top_k=5, top_p=0.9, filter_value=-float('Inf')):
    """
    :param logits: (vocab_size, )
    :param top_k:  取前topk个,
    :param top_p: 取前top概率为top_p的结果,当两者任一先满足时,则停止
    :param filter_value:
    :return:
    """
    assert logits.dim() == 1
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # torch.topk()返回最后一维最大的top_k个元素，返回值为二维(values,indices),values 和indices都是一维张量
        # ...表示其他维度由计算机自行推断, values返回结果由高到低排列,此处-1指的是取出最小的哪个概率,None是为了能广播.
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value  # 让概率为极小值,就不可能取到了

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)  # torch.cumsum()累加和
        # 排除后面累计概率大于限制的值
        sorted_indices_to_remove = cumulative_probs > top_p
        # 排除第一个值概率极大的情况
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        # 得到要放弃的值的原始列表的idxs
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


class Animator():
    def __init__(self, line_names=['train loss', 'train acc', 'test acc']) -> None:
        self.lines_dic = {}
        self.lines_name = line_names

    def add_dynamic(self, *args):
        display.clear_output(wait=True)
        if 'line0' not in self.lines_dic:
            self.lines_dic['line0'] = []
            (self.lines_dic['line0']).append(args[0])
        else:
            (self.lines_dic['line0']).append(args[0])
        for i in range(1, len(args)):
            if 'line' + str(i) not in self.lines_dic:
                self.lines_dic['line' + str(i)] = []
                (self.lines_dic['line' + str(i)]).append(args[i])
            else:
                (self.lines_dic['line' + str(i)]).append(args[i])
            plt.plot(self.lines_dic['line0'], self.lines_dic['line' + str(i)], "*-", label=self.lines_name[i - 1])
        plt.xlabel('epochs')
        plt.ylabel('number')
        plt.legend(loc='best')
        plt.grid()
        plt.show()
        plt.pause(0.1)


class Accumulator:
    """For accumulating sums over `n` variables."""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def grad_clipping(net, theta):
    """Clip the gradient."""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


class Timer:
    """简易计时器"""
    def __init__(self):
        self.time = []
        self.start()

    def start(self):
        self.time.append(datetime.datetime.now())

    def stop(self):
        return (datetime.datetime.now() - self.time[-1]).total_seconds()
