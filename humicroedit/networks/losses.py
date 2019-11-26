from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence

__all__ = ['MSELoss', 'SoftCrossEntropyLoss']


def sequence_mean(packed_sequence, apply=lambda x: x, padding_value=-100):
    """Calculate the mean of each sequence.
    Args:
        packed_sequence: PackedSequence
    Return:
        mean: (N,)
    """
    values, lengths = pad_packed_sequence(packed_sequence,
                                          batch_first=True,
                                          padding_value=padding_value)
    values = apply(values)
    return torch.stack([
        value[:length].float().mean()
        for value, length in zip(values, lengths)
    ], dim=0)


class MSELoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, feed, **_):
        feed['x'], lengths = pad_packed_sequence(feed['x'], batch_first=True)
        assert (lengths == 1).all()

        feed['pred'] = feed['x'].flatten()
        feed['loss'] = F.mse_loss(feed['pred'], sequence_mean(feed['y']))
        return feed


class SoftCrossEntropyLoss(nn.Module):

    def __init__(self, num_classes=4):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, feed, **_):
        feed['x'], lengths = pad_packed_sequence(feed['x'], batch_first=True)
        assert (lengths == 1).all()

        feed['pred'] = feed['x'].softmax(dim=1) @ \
            torch.arange(self.num_classes).to(feed['x'].device).float()

        if self.training:
            y = pad_packed_sequence(feed['y'],
                                    batch_first=True,
                                    padding_value=self.num_classes)[0]

            eye = torch.eye(self.num_classes + 1, device=y.device)
            y = eye[y][..., :-1].sum(dim=1)  # to onehot
            y /= y.sum(dim=-1, keepdim=True)  # normalize

            feed['loss'] = (-y * feed['x'].log_softmax(1)).sum(1).mean(0)
        else:
            feed['loss'] = F.mse_loss(feed['pred'], sequence_mean(feed['y']))

        return feed
