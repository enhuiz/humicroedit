import random
import pandas as pd
from collections import Counter
from operator import itemgetter


class Vocab():
    specials = ['<pad>', '<unk>', '<s>', '</s>', '<mask>',
                '<cls>', '<sep>', '<swap1>', '<swap2>', '<swap3>']

    unk = specials[1]

    max_size = 10000

    @staticmethod
    def special2index(e):
        return Vocab.specials.index(e)

    def __init__(self, corpus: [[str]]):
        self.counter = Counter([w for s in corpus for w in s])

        for special in self.specials:
            del self.counter[special]

        most_common = map(itemgetter(0),
                          self.counter.most_common(self.max_size))

        self.words = sorted(set(most_common))
        self.itos = self.specials + self.words
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def token2index(self, word):
        if word not in self.stoi:
            word = self.unk
        return self.stoi[word]

    def index2token(self, index):
        assert index < len(self), 'but get {}'.format(index)
        return self.itos[index]

    def indices2tokens(self, indices):
        return list(map(self.index2token, indices))

    def tokens2indices(self, words):
        return list(map(self.token2index, words))

    def __len__(self):
        return len(self.itos)

    def __str__(self):
        return ("Corpus(#words={})\n"
                "Vocab(#total={}, #words={}, #specials={})\n"
                "Sample words: {}").format(len(self.counter),
                                           len(self),
                                           len(self.itos) - len(self.specials),
                                           len(self.specials),
                                           random.sample(self.itos, 3))

    def __iter__(self):
        for s in self.itos:
            yield s

    def load(self, path):
        # a stupid title called "Null"
        df = pd.read_csv(path, na_filter=False)
        sentences = df['text'].tolist()
        sentences = map(str.split, sentences)
        self.__init__(sentences)
        print('Vocab is built on corpus: {}'.format(path))
        print(self)


vocab = Vocab([])
