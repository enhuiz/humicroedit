#!/usr/bin/env python3

import os
import re
import numpy as np
import argparse
import pandas as pd
import tqdm
import spacy
from pandarallel import pandarallel

nlp = spacy.load('en')

pandarallel.initialize(progress_bar=True)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='data/examiner')
    parser.add_argument('--num_samples', type=int, default=300000)
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
    # process the sentences
    df['text'] = df['text'].parallel_apply(process_sentence)
    return df


def main():
    args = get_args()

    path = os.path.join(args.root, 'examiner-date-text.csv')
    df = pd.read_csv(path)

    del df['publish_date']
    df.columns = ['text']

    df = process(df.head(args.num_samples))
    outpath = path.replace('.csv', '.preprocessed.csv')

    df.to_csv(outpath, index=None)


if __name__ == "__main__":
    main()
