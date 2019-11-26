import torch.nn as nn
import torch.nn.functional as F

from humicroedit.networks.layers import XApplier, XYApplier, TemporalPooling, Serial, PrependCLS
from humicroedit.networks.encoders import LSTMEncoder, TransformerEncoder
from humicroedit.networks.losses import MSELoss, SoftCrossEntropyLoss
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
                XApplier(TemporalPooling(lambda x: x[..., 0])),
                XApplier(nn.Linear(dim, 1)),
                MSELoss(),
            )
        elif name == 'baseline-ce':
            model = Serial(
                XApplier(PrependCLS()),
                XApplier(nn.Embedding(vocab_size, dim), broadcast=True),
                XApplier(LSTMEncoder(num_layers, dim)),
                XApplier(TemporalPooling(lambda x: x[..., 0])),
                XApplier(nn.Linear(dim, 4)),  # 4 for 4 different grades
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
                XApplier(TemporalPooling(lambda x: x[..., 0])),
                XApplier(nn.Linear(dim, 1)),
                MSELoss(),
            )
        elif name == 'baseline-ce':
            model = Serial(
                XApplier(PrependCLS()),
                XApplier(nn.Embedding(vocab_size, dim), broadcast=True),
                TransformerEncoder(num_layers, num_heads, dim, rpe_k=4),
                XApplier(TemporalPooling(lambda x: x[..., 0])),
                XApplier(nn.Linear(dim, 4)),
                SoftCrossEntropyLoss(),
            )
    else:
        raise Exception("Unknown model: {}.".format(name))

    return model
