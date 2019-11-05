import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence

__all__ = ['LSTMEncoder']


class LSTMEncoder(nn.Module):
    def __init__(self, num_layers, dim, bidirectional=True):
        super().__init__()
        self.layers = nn.ModuleList([nn.LSTM(dim,
                                             dim // (bidirectional + 1),
                                             num_layers=1,
                                             bidirectional=bidirectional)
                                     for _ in range(num_layers)])

    def forward(self, inputs):
        """
        Args:
            inputs: packed sequence
        """
        x = inputs
        for layer in self.layers:
            identity = x
            x = layer(x)[0]
            x = (PackedSequence(x.data + identity.data,  # residue
                                inputs.batch_sizes,
                                inputs.sorted_indices,
                                inputs.unsorted_indices)
                 .to(device=x.data.device))
        return x
