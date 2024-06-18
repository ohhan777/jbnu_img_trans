import os
import torch.nn as nn

RANK = int(os.getenv('RANK', -1))

def is_main_process():
    return (RANK in [-1, 0])

def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)

def de_parallel(model):
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model