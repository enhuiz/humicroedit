import torch.nn as nn

from humicroedit.networks.layers import XWrapper, HumorAttention, Serial
from humicroedit.networks.encoders import LSTMEncoder
from humicroedit.networks.losses import MSELoss


def get(name):
    dim = 512
    vocab_size = 5005

    if name == 'baseline':
        model = Serial(
            XWrapper(nn.Embedding(vocab_size, dim), unpack=True),
            XWrapper(LSTMEncoder(2, dim)),
            XWrapper(HumorAttention(dim)),
            XWrapper(nn.Linear(dim, 1)),
            MSELoss(),
        )
    else:
        raise Exception("Unknown model: {}.".format(name))

    return model
