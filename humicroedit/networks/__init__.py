import os
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence

from humicroedit.networks.layers import Applier, Serial, PrependCLS, RandomMask, SelectCLS, SelectMasked, Parallel, Lambda, TokenSegmentEmbedding
from humicroedit.networks.encoders import LSTMEncoder, TransformerEncoder
from humicroedit.networks.losses import MSELoss, SoftCrossEntropyLoss
from humicroedit.datasets.vocab import Vocab


def get(name):
    dim = 128
    num_layers = 4
    num_segments = 70
    vocab_size = len(Vocab.specials) + Vocab.max_size

    framework, contextual, loss = \
        name.split(os.path.sep)[1].split('-')[:3]

    # select contextural feature extractor, lstm or transformer
    if contextual == 'lstm':
        contextual_layers = [
            Applier(LSTMEncoder(num_layers, dim)),
        ]
    elif contextual == 'transformer':
        num_heads = 4
        rpe_k = 0
        contextual_layers = [
            TransformerEncoder(num_layers, num_heads, dim, rpe_k=rpe_k),
        ]
    else:
        raise Exception("Unknown contextual feature extractor: {}."
                        .format(contextual))

    # select loss, soft cross entropy vs mse
    if loss == 'sce':
        loss_layers = [
            Applier(nn.Linear(dim, 4)),
            SoftCrossEntropyLoss(),
        ]
    elif loss == 'mse':
        loss_layers = [
            Applier(nn.Linear(dim, 1)),
            MSELoss(),
        ]
    else:
        raise Exception("Unknown loss: {}.".format(loss))

    # select framework, the baseline framework or the bert framework
    if framework == 'baseline':
        model = Serial(
            PrependCLS(),
            TokenSegmentEmbedding(dim, vocab_size, num_segments),
            *contextual_layers,
            # select CLS only
            SelectCLS(),
            *loss_layers,
        )
    elif framework == 'bert':
        p_mask = 0.15
        model = Serial(
            PrependCLS(),
            RandomMask(p_mask),
            TokenSegmentEmbedding(dim, vocab_size, num_segments),
            *contextual_layers,
            Parallel(
                Serial(
                    SelectCLS(),
                    *loss_layers,
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
                'id': feeds[0]['id'],
                'pred': feeds[0]['pred'],
                'x': feeds[0]['x'],
                'loss': torch.stack([feeds[0]['loss'],
                                     feeds[1]['loss'] / math.log(vocab_size)]),
            }),
        )
    else:
        raise Exception("Unknown framework: {}.".format(framework))

    return model
