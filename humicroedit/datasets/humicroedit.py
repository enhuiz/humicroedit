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


def extract_edited(s):
    return re.sub(r'<swap1>.+<swap2> (.+) <swap3>', r'\1', s)


def extract_original(s):
    return re.sub(r'<swap1> (.+) <swap2>.+<swap3>', r'\1', s)


def kg_split(s):
    # remove the first '' since <kg-*> appears at the first
    return re.split(r'<kg-.+?>', s.strip())[1:]


def text_assemble(row, use_kg):
    if use_kg:
        text = ' <sep> '.join([
            extract_original(row['text']),
            *kg_split(row['org_kg']),
            extract_edited(row['text']),
            *kg_split(row['edt_kg'])
        ])
        assert len(text.split('<sep>')) == 20
    else:
        text = ' <sep> '.join([
            extract_original(row['text']),
            extract_edited(row['text']),
        ])
        assert len(text.split('<sep>')) == 2
    return text


@lru_cache()
def load_corpus(root, split, use_kg=False):
    filename = '{}.preprocessed{}.csv'.format(
        split, '.kg.processed' if use_kg else '')

    path = os.path.join(root, filename)
    df = pd.read_csv(path)

    df['text'] = df.apply(partial(text_assemble, use_kg=use_kg), axis=1)

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

    def __init__(self, root, split, use_kg=False):
        self.root = root
        self.split = split.replace('-small', '')
        self.training = 'train' in split
        self.use_kg = use_kg
        self.vocab = build_vocab(self.root)
        self.small = 'small' in split
        self.make_samples()
        print(self.vocab)

    def load_corpus(self):
        return load_corpus(self.root, self.split, self.use_kg)

    def make_samples(self):
        df = load_corpus(self.root, self.split, self.use_kg)
        if self.small:
            df = df.head(500)
        self.samples = df[['id', 'text', 'grade']].values

    def __getitem__(self, index):
        id_, sentence, grades = self.samples[index]
        tokens = sentence.strip().split()
        indices = self.vocab.tokens2indices(tokens)

        return {
            'id': id_,
            'tokens': tokens,
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
        return '{}\nSamples: {}'.format(self.vocab, [
            [
                item
                for sample in self.samples[:2]
                for item in sample
            ]
        ])
