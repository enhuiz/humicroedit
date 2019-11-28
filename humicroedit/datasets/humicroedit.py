import os
import re
import numpy as np
import pandas as pd
from functools import partial, lru_cache
from collections import defaultdict

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_sequence, pad_sequence

from humicroedit.datasets.vocab import vocab

import spacy
nlp = spacy.load('en')


def extract_edited(s):
    return re.sub(r'<swap1>.+<swap2> (.+) <swap3>', r'\1', s)


def extract_original(s):
    return re.sub(r'<swap1> (.+) <swap2>.+<swap3>', r'\1', s)


def kg_split(s):
    # remove the first '' since <kg-*> appears at the first
    s = re.sub(r'<kg-(.+?)>', r'<kg> \1', s.strip())
    ss = re.split(r'<kg>', s.strip())[1:]
    return ss


def text_assemble(row, use_kg):
    if use_kg:
        text = ' <sep> '.join([
            extract_original(row['text']),
            *kg_split(row['org_kg']),
            extract_edited(row['text']),
            *kg_split(row['edt_kg'])
        ])
        assert len(text.split('<sep>')) == 20 or\
            len(text.split('<sep>')) == 70
    else:
        text = ' <sep> '.join([
            extract_original(row['text']),
            extract_edited(row['text']),
        ])
        assert len(text.split('<sep>')) == 2
    return text


@lru_cache()
def load_corpus(root, split, kg_type=None):
    kg_suffix = '.{}.processed'.format(kg_type) if kg_type else ''

    path = os.path.join(root, '{}.preprocessed{}.csv'.format(split, kg_suffix))
    df = pd.read_csv(path)

    assembler = partial(text_assemble, use_kg=kg_type is not None)
    df['text'] = df.apply(assembler, axis=1)

    if 'grades' in df.columns:
        df['grade'] = df['grades'].apply(lambda s: list(map(int, str(s))))
    else:
        df['grade'] = df['text'].apply(lambda _: [np.nan])

    return df


class HumicroeditDataset(Dataset):

    def __init__(self, root, split, kg_type=None):
        self.root = root
        self.split = split.replace('-small', '')
        self.training = 'train' in split
        self.kg_type = kg_type
        self.small = 'small' in split
        self.make_samples()
        print(self)

    def load_corpus(self):
        df = load_corpus(self.root, self.split, self.kg_type)
        if self.small:
            df = df.head(500)
        return df

    def make_samples(self):
        df = self.load_corpus()
        self.samples = df[['id', 'text', 'grade']].values

    def __getitem__(self, index):
        id_, sentence, grades = self.samples[index]
        tokens = sentence.strip().split()
        indices = vocab.tokens2indices(tokens)

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
        return 'Samples:\n{}'.format('\n'.join([
            str(self.__getitem__(i)) for i in range(2)
        ]))
