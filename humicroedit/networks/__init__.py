from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence

from humicroedit.networks.layers import XApplier, XYApplier, Serial, PrependCLS, RandomMask, TemporalSelection
from humicroedit.networks.encoders import LSTMEncoder, TransformerEncoder
from humicroedit.networks.losses import MSELoss, SoftCrossEntropyLoss, BERTCrossEntropyLoss
from humicroedit.datasets.vocab import Vocab


def get(name):
    dim = 512
    num_layers = 4
    vocab_size = len(Vocab.specials) + 10000

    if 'lstm-' in name:
        name = name.replace('lstm-', '')
        if name == 'baseline':
            model = Serial(
                XApplier(PrependCLS()),
                XApplier(nn.Embedding(vocab_size, dim), broadcast=True),
                XApplier(LSTMEncoder(num_layers, dim)),
                # select CLS only
                XApplier(TemporalSelection(lambda x, _: x[..., :1])),
                XApplier(nn.Linear(dim, 1), broadcast=True),
                MSELoss(),
            )
        elif name == 'baseline-ce':
            model = Serial(
                XApplier(PrependCLS()),
                XApplier(nn.Embedding(vocab_size, dim), broadcast=True),
                XApplier(LSTMEncoder(num_layers, dim)),
                # select CLS only
                XApplier(TemporalSelection(lambda x, _: x[..., :1])),
                # 4 for 4 different grades
                XApplier(nn.Linear(dim, 4), broadcast=True),
                SoftCrossEntropyLoss(),
            )
    elif 'transformer-' in name:
        name = name.replace('transformer-', '')
        num_heads = 8
        if name == 'baseline':
            model = Serial(
                XApplier(PrependCLS()),
                XApplier(nn.Embedding(vocab_size, dim), broadcast=True),
                TransformerEncoder(num_layers, num_heads, dim, rpe_k=4),
                # select CLS only
                XApplier(TemporalSelection(lambda x, _: x[..., :1])),
                XApplier(nn.Linear(dim, 1), broadcast=True),
                MSELoss(),
            )
        elif name == 'baseline-ce':
            model = Serial(
                XApplier(PrependCLS()),
                XApplier(nn.Embedding(vocab_size, dim), broadcast=True),
                TransformerEncoder(num_layers, num_heads, dim, rpe_k=4),
                # select CLS only
                XApplier(TemporalSelection(lambda x, _: x[..., :1])),
                XApplier(nn.Linear(dim, 4), broadcast=True),
                SoftCrossEntropyLoss(),
            )
        elif name == 'baseline-pretraining':
            p_mask = 0.15

            model = Serial(
                RandomMask(p_mask),
                XApplier(nn.Embedding(vocab_size, dim), broadcast=True),
                TransformerEncoder(num_layers, num_heads, dim, rpe_k=4),
                XApplier(nn.Linear(dim, vocab_size), broadcast=True),
                XYApplier(nn.CrossEntropyLoss(), 'loss', broadcast=True),
            )
    else:
        raise Exception("Unknown model: {}.".format(name))

    return model
