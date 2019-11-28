import os
import re
import numpy as np
import pandas as pd
from functools import partial, lru_cache
from collections import defaultdict

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_sequence, pad_sequence


from .humicroedit import HumicroeditDataset


class ToyDataset(HumicroeditDataset):
    def __init__(self):
        self.make_samples()
        print(self)

    def load_corpus(self):
        numbers = [np.random.randint(0, 1000000) * 10 for i in range(100000)]
        df = pd.DataFrame({
            'text': [' '.join(str(number)) for number in numbers],
            # grade is the average of the number smaller than 4
            'grade': [[int(x) for x in str(number) if int(x) < 4]
                      for number in numbers]
        })
        df['id'] = df['grade'].apply(np.mean)
        return df
