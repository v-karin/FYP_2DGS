import torch
from torch import Tensor


def extend_dim(x: Tensor, front: int=0, back: int=0):
    length = len(x.shape)
    ext_front = max(front - length, 0)
    ext_back = max(back - ext_front - length, 0)
    return x.reshape((1,) * ext_front + x.shape + (1,) * ext_back)


def new_rot_mat(th: Tensor):
    sin_th, cos_th = th.sin(), th.cos()
    return torch.stack([
        torch.stack([cos_th, -sin_th], -1),
        torch.stack([sin_th, cos_th], -1)
    ], -1)
