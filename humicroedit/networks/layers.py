import copy
import random
import math
import logging
from functools import partial
from collections import defaultdict, OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence

from humicroedit import networks


def padding_mask(lens):
    """Mask out the blank (padding) values
    Args:
        lens: (bs,)
    Returns:
        mask: (bs, 1, max_len)
    """
    bs, max_len = len(lens), max(lens)
    mask = torch.zeros(bs, 1, max_len)
    for i, l in enumerate(lens):
        mask[i, :, :l] = 1
    mask = mask > 0
    return mask


class HumorAttention(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.dim = dim
        # the essential of humor :)
        self.humor = nn.Parameter(torch.randn(1, dim))
        self.k_linear = nn.Linear(dim, dim)
        self.v_linear = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: query (*, len, dim)
        Returns:
            context: (*, dim)
        """
        x, lengths = pad_packed_sequence(x, True)
        mask = padding_mask(lengths).to(x.device)

        k = self.k_linear(x)
        v = self.v_linear(x)

        q = self.humor / self.dim ** 0.5
        score = q @ k.transpose(-2, -1)

        score = score.masked_fill(mask == 0, -np.inf)

        alpha = F.softmax(score, dim=-1)

        # if one query is totally masked
        summation = alpha.sum(dim=-1)
        anynan = (summation != summation)[..., None]
        alpha = alpha.masked_fill(anynan, 0)

        alpha = self.dropout(alpha)
        context = alpha @ v

        return context.squeeze(1)


class XWrapper(nn.Module):
    def __init__(self, layer, unpack=False):
        super().__init__()
        self.layer = layer
        self.unpack = unpack

    def forward(self, feed, **_):
        if self.unpack:
            data = feed['x'].data
            data = self.layer(data)
            feed['x'] = (PackedSequence(data,
                                        feed['x'].batch_sizes,
                                        feed['x'].sorted_indices,
                                        feed['x'].unsorted_indices)
                         .to(device=data.device))
        else:
            feed['x'] = self.layer(feed['x'])
        return feed


class Serial(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self._device = nn.Parameter()
        self.layers = nn.ModuleList(layers)

    @property
    def device(self):
        return self._device.device

    def forward(self, feed, *args, **kwargs):
        # pytorch bug: https://github.com/pytorch/pytorch/issues/22251
        # device= is neccessary
        feed['x'] = feed['x'].to(device=self.device)
        feed['y'] = feed['y'].to(device=self.device)
        for layer in self.layers:
            feed = layer(feed, *args, **kwargs)
        return feed


class Parallel(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, feed, *args, **kwargs):
        return [layer(feed.copy(), *args, **kwargs) for layer in self.layers]
