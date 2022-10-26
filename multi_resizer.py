from resizer import Resizer
from torch import nn
import torch

class MultiResizer(nn.Module):
    def __init__(self, N_mask, shape, device):
        super(MultiResizer, self).__init__()
        shape_d = list(shape)
        down_Ns = torch.unique(N_mask)
        self.N_mask = N_mask
        self.up = {}
        self.down = {}
        for down_N in down_Ns:
            shape_d[2] = int(shape[2] / down_N.item())
            shape_d[3] = int(shape[3] / down_N.item())
            self.down[down_N] = Resizer(shape, 1 / down_N.item()).to(device)
            self.up[down_N] = Resizer(tuple(shape_d), down_N.item()).to(device)

    def forward(self, in_tensor, *ignore_args, **ignore_kwargs):
        out_tensor = torch.zeros(in_tensor.shape).to(in_tensor.device)
        for down_N in self.down.keys():
            out_tensor = out_tensor + (self.N_mask == down_N) * self.up[down_N](self.down[down_N](in_tensor))
        return out_tensor
