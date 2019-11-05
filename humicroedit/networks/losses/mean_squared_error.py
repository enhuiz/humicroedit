import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence

__all__ = ['MSELoss']


class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feed, **_):
        feed['loss'] = F.mse_loss(feed['x'], feed['y'])
        return feed
