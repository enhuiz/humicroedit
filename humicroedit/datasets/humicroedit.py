import os
import re
import numpy as np
import pandas as pd
from functools import partial, lru_cache
from collections import defaultdict

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_sequence, pad_sequence

from humicroedit.datasets.vocab import Vocab


@lru_cache()
def load_corpus(root, split):
    path = os.path.join(root, '{}.preprocessed.csv'.format(split))
    df = pd.read_csv(path)

    if 'grades' in df.columns:
        df['grade'] = df['grades'].apply(lambda s: list(map(int, str(s))))
    else:
        df['grade'] = df['text'].apply(lambda _: [np.nan])

    return df


@lru_cache()
def build_vocab(root):
    """
    Build vocab for the task, only word in train will be used.
    """
    df = load_corpus(root, 'train')
    sentences = df['text'].tolist()
    sentences = map(str.split, sentences)
    vocab = Vocab(sentences)
    return vocab


class HumicroeditDataset(Dataset):
    ignore_index = -100

    def __init__(self, root, split, pretraining=False):
        self.root = root
        self.split = split
        self.training = 'train' in split
        self.pretraining = pretraining
        self.vocab = build_vocab(self.root)
        self.make_samples()

    def load_corpus(self):
        return load_corpus(self.root, self.split)

    @staticmethod
    def extract_original(s):
        return re.sub(r'<swap1> (.+) <swap2>.+<swap3>', r'\1', s)

    def make_samples(self):
        df = load_corpus(self.root, self.split)
        if self.pretraining:
            df['text'] = df['text'].apply(self.extract_original)
        self.samples = df[['id', 'text', 'grade']].values

    def __getitem__(self, index):
        id_, sentence, grades = self.samples[index]
        indices = self.vocab.tokens2indices(sentence.strip().split())
        return {
            'id': id_,
            'tokens': sentence.strip().split(),
            'indices': indices,
            'grades': grades,
        }

    def get_collate_fn(self):

        def collate_fn(batch):
            batch = sorted(batch, key=lambda s: -len(s['indices']))

            x = pack_sequence([torch.tensor(sample['indices'])
                               for sample in batch])

            y = pack_sequence([torch.tensor(sample['grades'])
                               for sample in batch],
                              enforce_sorted=False)

            return {
                'x': x,
                'y': y,
                'id': [sample['id'] for sample in batch],
                'token': [sample['tokens'] for sample in batch],
            }

        return collate_fn

    def __len__(self):
        return len(self.samples)

    def __str__(self):
        return '{}\nExamples: {}'.format(self.vocab, self.samples[:2])
