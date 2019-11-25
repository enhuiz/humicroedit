import torch.nn as nn

from humicroedit.networks.layers import XApplier, TemporalPooling, Serial, PrependCLS
from humicroedit.networks.encoders import LSTMEncoder, TransformerEncoder
from humicroedit.networks.losses import MSELoss


def get(name):
    dim = 128
    num_layers = 4
    vocab_size = 5005

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
    else:
        raise Exception("Unknown model: {}.".format(name))

    return model
