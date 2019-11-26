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
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence, pack_sequence

from humicroedit import networks
from humicroedit.datasets.vocab import Vocab


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None, rpe_q=None, rpe_v=None):
        """
        Args:
            q: query (*, query_len, dim)
            k: key (*, key_len, dim)
            v: value (*, key_len, dim)
            mask: (*, query_len, key_len), 0 will be masked out
            rpe_q : (query_len, key_len, dim)
            rpe_v : (query_len, key_len, dim)
        Returns:
            context: (*, query_len, dim)
            weights: (*, query_len, key_len)
        """
        dim = q.shape[-1]

        q /= dim ** 0.5
        score = q @ k.transpose(-2, -1)

        if rpe_q is not None:
            score += (q.unsqueeze(-2) @ rpe_q.transpose(-2, -1)).squeeze(-2)

        if mask is not None:
            score = score.masked_fill(mask == 0, -np.inf)

        alpha = F.softmax(score, dim=-1)

        summation = alpha.sum(dim=-1)
        anynan = (summation != summation)[..., None]
        alpha = alpha.masked_fill(anynan, 0)

        alpha = self.dropout(alpha)
        context = alpha @ v

        if rpe_v is not None:
            context += (alpha.unsqueeze(-2) @ rpe_v).squeeze(-2)

        return context, alpha


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, dim, dropout, rpe_k=0,
                 attention=ScaledDotProductAttention):
        assert dim % heads == 0, "dim should be a multiple of heads, \
            got {} and {}".format(dim, heads)

        super().__init__()

        self.heads = heads
        self.dim = dim

        self.q_linear = nn.Linear(dim, dim)
        self.k_linear = nn.Linear(dim, dim)
        self.v_linear = nn.Linear(dim, dim)

        self.rpe_k = rpe_k
        if rpe_k > 0:
            self.rpe_w = nn.Embedding(rpe_k * 2 + 1, 2 * dim // heads)
            logging.info('Using rpe_k={}.'.format(rpe_k))

        self.attention = attention(dropout)
        self.fc = nn.Linear(dim, dim)

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q: query (batch, query_len, dim)
            k: key (batch, key_len, dim)
            v: value (batch, query_len, dim)
            mask: (batch, query_len, key_len)
        Returns:
            context: (batch, query_len, dim)
        """
        bs, ql = q.shape[:2]

        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)

        q = q.view(bs, -1, self.heads, self.dim // self.heads)
        k = k.view(bs, -1, self.heads, self.dim // self.heads)
        v = v.view(bs, -1, self.heads, self.dim // self.heads)

        # swap len and head
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # add head dim for mask
        if mask is not None:
            mask = mask.unsqueeze(1)

        if self.rpe_k > 0:
            indices = torch.arange(q.shape[-2])
            indices = indices.repeat(q.shape[-2], 1)
            distance = indices - indices.transpose(0, 1)
            distance = torch.clamp(distance, -self.rpe_k, self.rpe_k)
            indices = distance + self.rpe_k
            indices = indices.to(q.device)
            rpe = self.rpe_w(indices)
            rpe_q, rpe_v = rpe.chunk(2, dim=-1)
            context, alpha = self.attention(q, k, v, mask, rpe_q, rpe_v)
        else:
            context, alpha = self.attention(q, k, v, mask)

        self.alpha = alpha.detach().cpu()  # for vis

        # swap len and head back
        context = context.transpose(1, 2)
        context = context.reshape(bs, ql, self.dim)
        context = self.fc(context)

        return context


class PositionwiseFeedForward(nn.Module):
    def __init__(self, dim, ffn_dim, dropout):
        super().__init__()
        self.dim = dim
        self.w1 = nn.Linear(dim, ffn_dim)
        self.w2 = nn.Linear(ffn_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.dropout(F.relu(self.w1(x))))


class SublayerConnection(nn.Module):
    def __init__(self, sublayer, dropout):
        super().__init__()
        self.sublayer = sublayer
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(sublayer.dim)

    def forward(self, x, applier=lambda x: (x,)):
        return x + self.dropout(self.sublayer(*applier(self.norm(x))))


class TemporalSelection(nn.Module):
    def __init__(self, select):
        super().__init__()
        self.select = select

    def forward(self, x):
        values, lengths = pad_packed_sequence(x, True)
        packed = pack_sequence([
            (self.select(value[None, :length].permute(0, 2, 1), i)
             .permute(0, 2, 1)
             .squeeze(0))
            for i, (value, length) in enumerate(zip(values, lengths))
        ], enforce_sorted=False)
        return packed


class PrependCLS(nn.Module):
    cls = Vocab.special2index('<cls>')

    def __init__(self):
        super().__init__()

    def forward(self, x):
        values, lengths = pad_packed_sequence(x, True)
        x = [torch.cat([torch.tensor([self.cls]).to(value.device),
                        value[:length]])
             for value, length in zip(values, lengths)]
        x = pack_sequence(x, False)
        return x


class RandomMask(nn.Module):
    mask_index = Vocab.special2index('<mask>')
    ignore_index = -100

    def __init__(self, p_mask=0.15):
        super().__init__()
        self.p_mask = p_mask

    def forward(self, feed):
        feed['y'] = feed['x']

        x, lengths = pad_packed_sequence(feed['x'], True)

        mask = pack_sequence([torch.rand(length).to(x.device) < self.p_mask
                              for length in lengths], enforce_sorted=False)

        feed['x'] = (PackedSequence(
            feed['x'].data.masked_fill(mask.data, self.mask_index),
            feed['x'].batch_sizes,
            feed['x'].sorted_indices,
            feed['x'].unsorted_indices
        ).to(device=x.device))

        feed['y'] = (PackedSequence(
            feed['y'].data.masked_fill(~mask.data, self.ignore_index),
            feed['y'].batch_sizes,
            feed['y'].sorted_indices,
            feed['y'].unsorted_indices
        ).to(device=x.device))

        return feed


class XApplier(nn.Module):
    def __init__(self, layer, broadcast=False, key_out='x'):
        super().__init__()
        self.layer = layer
        self.broadcast = broadcast
        self.key_out = key_out

    def forward(self, feed, **_):
        if self.broadcast:
            data = feed['x'].data
            data = self.layer(data)
            feed[self.key_out] = (PackedSequence(data,
                                                 feed['x'].batch_sizes,
                                                 feed['x'].sorted_indices,
                                                 feed['x'].unsorted_indices)
                                  .to(device=data.device))
        else:
            feed[self.key_out] = self.layer(feed['x'])
        return feed


class XYApplier(nn.Module):
    def __init__(self, layer, key_out, broadcast):
        super().__init__()
        self.layer = layer
        self.key_out = key_out
        self.broadcast = broadcast

    def forward(self, feed, **_):
        if self.broadcast:
            feed[self.key_out] = self.layer(feed['x'].data, feed['y'].data)
        else:
            feed[self.key_out] = self.layer(feed['x'], feed['y'])
        return feed


class Lambda(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, feed):
        return self.lambd(feed)


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
