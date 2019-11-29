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

pandarallel.initialize(nb_workers=8, progress_bar=True)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='data/humicroedit/task-1')
    parser.add_argument('--kg-type', type=str, default='kgc')
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


def convert(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1 \2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1 \2', s1).lower()


def tokenize(col):
    col = col.replace('original-', '').replace('edited-', '')
    col = convert(col)
    return '<kg-{}>'.format(col.lower())


def next_valid_sentence(sentences):
    iterator = (s for s in sentences if s != 'none' and s != "")
    return next(iterator, 'none')


def make_processor(cols):
    def processor(row):
        text = []
        for col in cols:
            s = next_valid_sentence(row[col])
            text.append(tokenize(col) + ' ' + process_sentence(s))
        text = ' '.join(text)
        return text
    return processor


def process(df):
    org_cols = [col for col in df.columns if 'original-' in col]
    edt_cols = [col for col in df.columns if 'edited-' in col]

    for col in org_cols + edt_cols:
        df[col] = df[col].apply(eval)

    df['org_kg'] = df.parallel_apply(make_processor(org_cols), axis=1)
    df['edt_kg'] = df.parallel_apply(make_processor(edt_cols), axis=1)

    for col in org_cols + edt_cols:
        del df[col]

    return df


def main():
    args = get_args()

    for split in ['train', 'dev']:
        path = os.path.join(args.root,
                            '{}.preprocessed.{}.csv'
                            .format(split, args.kg_type))
        print('processing', path)
        df = pd.read_csv(path)
        df = df.fillna('[]')
        df = process(df)
        outpath = path.replace('csv', 'processed.csv')
        df.to_csv(outpath, index=None)
        print(df.head()['text'])


if __name__ == "__main__":
    main()
