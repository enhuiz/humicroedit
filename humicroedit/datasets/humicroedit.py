import os
import re
import numpy as np
import pandas as pd
from functools import partial, lru_cache
from collections import defaultdict

import torch
from torch.nn.utils.rnn import pack_sequence
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_sequence

from humicroedit.datasets.vocab import Vocab

from nltk.tokenize import word_tokenize


def process_sentence(s):
    # make all characters lower case
    s = s.strip().lower()
    # convert year-old to year old, so that there is smaller vocab
    s = s.replace('-', ' ')
    # tokenize, convert he's to he 's
    s = ' '.join(word_tokenize(s))
    # convert number to digits, 123 -> 1 2 3,
    s = re.sub(r"([0-9])", r" \1 ", s).strip()
    # replace % with percent
    s = s.replace('%', 'percent')
    return s


@lru_cache()
def load_corpus(root, split):
    path = os.path.join(root, '{}.csv'.format(split))
    df = pd.read_csv(path)

    # substitute with the edit word.
    df['edited'] = df.apply(lambda row: re.sub(r'<.+?/>',
                                               row['edit'],
                                               row['original']),
                            axis=1)

    df['original'] = df.apply(lambda row: re.sub(r'<(.+?)/>',
                                                 r'\1',
                                                 row['original']),
                              axis=1)

    # process the sentences
    df['original'] = df['original'].apply(process_sentence)
    df['edited'] = df['edited'].apply(process_sentence)

    if 'meanGrade' not in df:
        df['meanGrade'] = np.nan

    return df


@lru_cache()
def build_vocab(root):
    """
    Build vocab for the task, only word in train will be used.
    """
    df = load_corpus(root, 'train')
    sentences = df['original'].tolist() + df['edited'].tolist()
    sentences = map(str.split, sentences)
    vocab = Vocab(sentences)
    return vocab


def interleave(*args):
    """interleave: [1, 2, 3], [4, 5, 6] |-> [1, 4, 2, 5, 3, 6]
    """
    return [x for l in zip(*args) for x in l]


class HumicroeditDataset(Dataset):
    def __init__(self, root, split):
        self.root = root
        self.split = split
        self.vocab = build_vocab(self.root)
        self.make_samples()

    def make_samples(self):
        df = load_corpus(self.root, self.split)

        odf = df[['id', 'original', 'meanGrade']].copy()
        odf['meanGrade'] = 0
        original_samples = odf.values.tolist()

        edf = df[['id', 'edited', 'meanGrade']].copy()
        edited_samples = edf.values.tolist()

        assert len(original_samples) == len(edited_samples)

        if 'train' in self.split:
            self.samples = interleave(original_samples, edited_samples)
        else:
            self.samples = edited_samples

    def __getitem__(self, index):
        id_, sentence, grade = self.samples[index]
        sentence = self.vocab.tokens2indices(sentence.strip().split())
        sentence = torch.tensor(sentence).long()
        return id_, sentence, grade

    def get_collate_fn(self):
        def collate_fn(batch):
            batch = sorted(batch, key=lambda s: -len(s[1]))
            ids = [sample[0] for sample in batch]
            sentences = pack_sequence([sample[1] for sample in batch])
            grades = torch.tensor([sample[2] for sample in batch])[:, None]
            batch = {
                'id': ids,
                'x': sentences,
                'y': grades
            }
            return batch
        return collate_fn

    def __len__(self):
        return len(self.samples)

    def __str__(self):
        return '{}\nExamples: {}'.format(self.vocab, self.samples[:2])
