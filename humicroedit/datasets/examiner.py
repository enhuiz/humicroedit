import os
import numpy as np
import pandas as pd
from functools import lru_cache

from humicroedit.datasets.humicroedit import HumicroeditDataset


@lru_cache()
def load_corpus(root):
    path = os.path.join(root, 'examiner-date-text.preprocessed.csv')
    df = pd.read_csv(path, na_filter=False)
    df['grade'] = df['text'].apply(lambda _: [0])
    df['id'] = df.index
    return df


class ExaminerDataset(HumicroeditDataset):
    def __init__(self, root):
        self.root = root
        self.make_samples()

    def load_corpus(self):
        return load_corpus(self.root)
