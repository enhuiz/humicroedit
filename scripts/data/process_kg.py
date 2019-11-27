#!/usr/bin/env python3

import os
import re
import numpy as np
import argparse
import pandas as pd
import tqdm
import spacy

nlp = spacy.load('en')

from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)


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
    s = ' '.join(s.split(' '))
    return s


def process(df):
    cols = [col for col in df.columns
            if 'original-' in col or 'edited-' in col]

    def tokenize(col):
        return '<{}>'.format(col.lower())

    def process_row(row):
        text = [row['text']]
        for col in cols:
            sentences = eval(row[col])
            s = next((s for s in sentences if s != 'none' and s != ""), 'none')
            text.append(tokenize(col) + ' ' + process_sentence(s))
        text = ' '.join(text)
        return text

    df['text'] = df.parallel_apply(process_row, axis=1)

    for col in cols:
        del df[col]

    return df


def main():
    args = get_args()

    for split in ['train', 'dev']:
        path = os.path.join(args.root,
                            '{}.preprocessed.kg.csv'.format(split))
        df = pd.read_csv(path)
        df = process(df)
        outpath = os.path.join(args.root,
                               split + '.preprocessed.kg.processed.csv')
        df.to_csv(outpath, index=None)
        print(df.head()['text'])


if __name__ == "__main__":
    main()
