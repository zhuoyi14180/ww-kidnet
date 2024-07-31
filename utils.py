import torch.distributed as dist
import logging
import numpy as np
import torch
import random


def setup(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    print(f"Process {rank}/{world_size} initialized on GPU {torch.cuda.current_device()}")

    return rank, world_size


def cleanup():
    dist.destroy_process_group()


def all_reduce_tensor(tensor, op=dist.ReduceOp.SUM, world_size=1):
    tensor = tensor.clone()
    dist.all_reduce(tensor, op)
    tensor.div_(world_size)
    return tensor


def log_args(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s ===> %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # args FileHandler to save log file
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # args StreamHandler to print log to console
    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(formatter)

    # add the two Handler
    logger.addHandler(sh)
    logger.addHandler(fh)


def adjust_learning_rate(optimizer, epoch, max_epoch, init_lr, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(init_lr * np.power(1 - (epoch) / max_epoch, power), 8)


class Accumulator:
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        """Defined in :numref:`sec_softmax_scratch`"""
        self.counter = 0
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
        self.counter += 1

    def reset(self):
        self.counter = 0
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    def avg(self):
        return [i/self.counter for i in self.data]


def one_hot(target, n_classes):
    B, H, W, D = target.size()
    res = torch.zeros((B, n_classes, H, W, D), dtype=target.dtype).cuda()
    for i in range(n_classes):
        index_list = (target == i).nonzero()

        for j in range(len(index_list)):
            batch, height, width, depth = index_list[j]
            res[B, i, H, W, D] = 1

    return res.float()