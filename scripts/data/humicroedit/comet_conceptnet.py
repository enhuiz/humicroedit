#!/usr/bin/env python3

import tqdm
import os
import sys
import argparse
import pandas as pd
import json
import re

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_packed_sequence

sys.path.append('.')
from humicroedit.utils import working_directory
from humicroedit.datasets.humicroedit import extract_original, extract_edited

# comet imports
sys.path.append('comet')
import src.data.data as data
import src.data.config as cfg
import src.interactive.functions as interactive


def disable_print():
    sys.stdout = open(os.devnull, 'w')


def enable_print():
    sys.stdout = sys.__stdout__


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default='data/humicroedit/task-1')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--category", type=str, default="all")
    parser.add_argument("--sampler", type=str, default="beam-3")
    args = parser.parse_args()
    args.root = os.path.abspath(args.root)
    return args


def extract_term(row, original=True):
    if original:
        term = row['text'].split('<swap1>')[1].split('<swap2>')[0].strip()
    else:
        term = row['text'].split('<swap2>')[1].split('<swap3>')[0].strip()
    return term


def generate(args, split):
    in_path = os.path.abspath(os.path.join(args.root,
                                           split + '.preprocessed.csv'))
    out_path = in_path.replace('.csv', '.kgc.csv')
    df = pd.read_csv(in_path)

    with working_directory('comet'):
        opt, state_dict = interactive.load_model_file(
            'pretrained_models/conceptnet_pretrained_model.pickle')

        data_loader, text_encoder = interactive.load_data("conceptnet", opt)

    n_ctx = data_loader.max_e1 + data_loader.max_e2 + data_loader.max_r
    n_vocab = len(text_encoder.encoder) + n_ctx
    model = interactive.make_model(opt, n_vocab, n_ctx, state_dict)

    if args.device != "cpu":
        cfg.device = args.device
        cfg.do_gpu = True
        model.to(cfg.device)
    else:
        cfg.device = "cpu"

    sampler = interactive.set_sampler(opt, args.sampler, data_loader)

    for i, row in tqdm.tqdm(list(df.iterrows())):
        # original sentence
        original = extract_term(row, True)
        disable_print()
        result = interactive.get_conceptnet_sequence(original,
                                                     model,
                                                     sampler,
                                                     data_loader,
                                                     text_encoder,
                                                     args.category)

        for key in result:
            df.at[i, 'original-' + key] = json.dumps(result[key]['beams'])

        # edited sentence
        edited = extract_term(row, False)
        result = interactive.get_conceptnet_sequence(edited,
                                                     model,
                                                     sampler,
                                                     data_loader,
                                                     text_encoder,
                                                     args.category)

        for key in result:
            df.at[i, 'edited-' + key] = json.dumps(result[key]['beams'])
        enable_print()

        df.to_csv(out_path, index=None)


def main():
    args = parse_args()

    for split in ['train', 'dev']:
        generate(args, split)


if __name__ == "__main__":
    main()
