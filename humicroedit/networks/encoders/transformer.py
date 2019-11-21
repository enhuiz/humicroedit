import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


from ..layers import SublayerConnection, MultiHeadAttention, PositionwiseFeedForward
from ..positional_encoding import PositionalEncoding
from ..utils import padding_mask

__all__ = ['TransformerEncoder']


class TransformerEncoderLayer(nn.Module):
    def __init__(self, self_attn, ffn, dropout):
        super().__init__()
        self.self_attn = SublayerConnection(self_attn, dropout)
        self.ffn = SublayerConnection(ffn, dropout)

    def forward(self, x, mask):
        """
        Args:
            x: (bs, src_len, dim)
            mask: (bs, src_len, src_len)
        """
        x = self.self_attn(x, lambda x: (x, x, x, mask))
        x = self.ffn(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, num_heads, dim,
                 ffn_dim=None, dropout=0.1, rpe_k=8):
        super().__init__()
        ffn_dim = dim * 4 if ffn_dim is None else ffn_dim

        self.norm = nn.LayerNorm(dim)

        def mha(i):
            return MultiHeadAttention(num_heads,
                                      dim,
                                      dropout,
                                      rpe_k)

        def ffn():
            return PositionwiseFeedForward(dim,
                                           ffn_dim,
                                           dropout)

        def layer(i):
            return TransformerEncoderLayer(mha(i), ffn(), dropout)

        self.layers = nn.ModuleList([layer(i) for i in range(num_layers)])

    def forward(self, feed, **_):
        x, length = pad_packed_sequence(feed['x'], batch_first=True)
        mask = padding_mask(length).to(x.device)

        for layer in self.layers:
            x = layer(x, mask)
        mem = self.norm(x)

        feed['x'] = pack_padded_sequence(mem, length,
                                         batch_first=True,
                                         enforce_sorted=False)
        return feed
