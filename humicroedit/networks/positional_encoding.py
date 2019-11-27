import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class PositionalEncoding(nn.Module):
    def __init__(self, dim, dropout=0.1, max_len=4096):
        assert dim % 2 == 0
        super().__init__()

        pos = np.arange(0, max_len)[:, None]
        i = np.arange(0, dim // 2)
        denom = 10000 ** (2 * i / dim)

        pe = np.zeros([max_len, dim])
        pe[:, 0::2] = np.sin(pos / denom)
        pe[:, 1::2] = np.cos(pos / denom)
        pe = torch.tensor(pe.tolist()).float()

        self.pe = nn.Parameter(pe)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.pe[:x.shape[1]]
        return self.dropout(x)
