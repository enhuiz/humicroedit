import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['MSELoss', 'CrossEntropyLoss']


class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feed, **_):
        feed['x'] = feed['x'].flatten()
        feed['loss'] = F.mse_loss(feed['x'], feed['y'])
        return feed


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feed, **_):
        x = feed['x'].softmax(dim=-1) @ \
            torch.arange(4).to(feed['x'].device).float()
        if self.training:
            feed['loss'] = F.cross_entropy(feed['x'], feed['y'])
        else:
            feed['loss'] = F.mse_loss(x, feed['y'].float())
        feed['x'] = x
        return feed
