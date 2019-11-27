#!/usr/bin/env python3

import os
import re
import numpy as np
import argparse
import pandas as pd
import tqdm
import spacy

nlp = spacy.load('en')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='data/humicroedit/task-1')
    args = parser.parse_args()
    return args


def process_sentence(s):
    # make all characters lower case
    s = s.strip().lower()
    # change ` to '
    s = s.replace('‘', "'")
    # change ‘ to '
    s = s.replace('’', "'")
    # remove comma inside number
    s = re.sub(r"(\d+?),", r"\1", s)
    # convert number to digits, 123 -> 1 2 3,
    s = re.sub(r"([0-9])", r" \1 ", s).strip()
    # lemmatization
    s = ' '.join([token.lemma_ for token in nlp(s)])
    # replace % with percent
    s = s.replace('%', 'percent')
    # remove extra space
    s = ' '.join(s.split())
    return s


def process(df):
    # substitute with the edit word.
    df['text'] = df.apply(lambda row: re.sub(r'<(.+?)/>',
                                             r'swapi \1 swapii {} swapiii'
                                             .format(row['edit']),
                                             row['original']),
                          axis=1)

    del df['original']

    # process the sentences
    df['text'] = df['text'].progress_apply(process_sentence)

    df['text'] = df['text'].apply(lambda s: s.replace('swapiii', '<swap3>')
                                  .replace('swapii', '<swap2>')
                                  .replace('swapi', '<swap1>'))

    return df


def main():
    args = get_args()
    tqdm.tqdm.pandas()
    for split in ['train', 'dev']:
        path = os.path.join(args.root, '{}.csv'.format(split))
        df = pd.read_csv(path)
        df = process(df)
        outpath = os.path.join(args.root, split + '.preprocessed.csv')
        df.to_csv(outpath, index=None)


if __name__ == "__main__":
    main()
