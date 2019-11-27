from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence

from humicroedit.networks.layers import Applier, Serial, PrependCLS, RandomMask, SelectCLS, SelectMasked, Parallel, Lambda
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
                PrependCLS(),
                Applier(nn.Embedding(vocab_size, dim), broadcast=True),
                Applier(LSTMEncoder(num_layers, dim)),
                # select CLS only
                SelectCLS(),
                Applier(nn.Linear(dim, 1)),
                MSELoss(),
            )
        elif name == 'baseline-ce':
            model = Serial(
                PrependCLS(),
                Applier(nn.Embedding(vocab_size, dim), broadcast=True),
                Applier(LSTMEncoder(num_layers, dim)),
                # select CLS only
                SelectCLS(),
                # 4 for 4 different grades
                Applier(nn.Linear(dim, 4)),
                SoftCrossEntropyLoss(),
            )
    elif 'transformer-' in name:
        name = name.replace('transformer-', '')
        num_heads = 8
        if name == 'baseline':
            model = Serial(
                PrependCLS(),
                Applier(nn.Embedding(vocab_size, dim), broadcast=True),
                TransformerEncoder(num_layers, num_heads, dim, rpe_k=4),
                # select CLS only
                SelectCLS(),
                Applier(nn.Linear(dim, 1)),
                MSELoss(),
            )
        elif name == 'baseline-ce':
            model = Serial(
                PrependCLS(),
                Applier(nn.Embedding(vocab_size, dim), broadcast=True),
                TransformerEncoder(num_layers, num_heads, dim, rpe_k=4),
                # select CLS only
                SelectCLS(),
                Applier(nn.Linear(dim, 4)),
                SoftCrossEntropyLoss(),
            )
        elif name == 'bert':
            p_mask = 0.15

            model = Serial(
                PrependCLS(),
                RandomMask(p_mask),
                Applier(nn.Embedding(vocab_size, dim), broadcast=True),
                TransformerEncoder(num_layers, num_heads, dim, rpe_k=4),
                Parallel(
                    Serial(
                        SelectCLS(),
                        Applier(nn.Linear(dim, 4)),
                        SoftCrossEntropyLoss(),
                    ),
                    Serial(
                        SelectMasked(),
                        Applier(nn.Linear(dim, vocab_size), broadcast=True),
                        Applier(nn.CrossEntropyLoss(),
                                k_in=['x', 'masked'],
                                k_out='loss',
                                broadcast=True),
                    )
                ),
                Lambda(lambda feeds: {
                    'pred': feeds[0]['pred'],
                    'x': feeds[0]['x'],
                    'loss': torch.stack([feeds[0]['loss'], feeds[1]['loss']]),
                }),
            )
    else:
        raise Exception("Unknown model: {}.".format(name))

    return model
