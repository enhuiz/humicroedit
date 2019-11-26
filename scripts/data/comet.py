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
from humicroedit.datasets.humicroedit import HumicroeditDataset

# comet imports
sys.path.append('comet')
import src.data.data as data
import src.data.config as cfg
import src.interactive.functions as interactive


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default='data/humicroedit/task-1')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--category", type=str, default="all")
    parser.add_argument("--sampler", type=str, default="beam-10")
    args = parser.parse_args()
    args.root = os.path.abspath(args.root)
    return args


def set_atomic_inputs(input_event, category, data_loader, text_encoder):
    XMB = torch.zeros(1, data_loader.max_event + 1).long().to(cfg.device)

    prefix, suffix = data.atomic_data.do_example(
        text_encoder, input_event, None, True, None)

    prefix = prefix[:data_loader.max_event]

    XMB[:, :len(prefix)] = torch.LongTensor(prefix)
    XMB[:, -1] = torch.LongTensor([
        text_encoder.encoder["<{}>".format(category)]
    ])

    batch = {}
    batch["sequences"] = XMB
    batch["attention_mask"] = data.atomic_data.make_attention_mask(XMB)

    return batch


def get_atomic_sequence(input_event, model, sampler, data_loader, text_encoder, category):
    if isinstance(category, list):
        outputs = {}
        for cat in category:
            new_outputs = get_atomic_sequence(
                input_event, model, sampler, data_loader, text_encoder, cat)
            outputs.update(new_outputs)
        return outputs
    elif category == "all":
        outputs = {}

        for category in data_loader.categories:
            new_outputs = get_atomic_sequence(
                input_event, model, sampler, data_loader, text_encoder, category)
            outputs.update(new_outputs)
        return outputs
    else:
        sequence_all = {}

        sequence_all["event"] = input_event
        sequence_all["effect_type"] = category

        with torch.no_grad():
            batch = set_atomic_inputs(
                input_event, category, data_loader, text_encoder)

            sampling_result = sampler.generate_sequence(
                batch, model, data_loader, data_loader.max_event +
                data.atomic_data.num_delimiter_tokens["category"],
                data_loader.max_effect -
                data.atomic_data.num_delimiter_tokens["category"])

        sequence_all['beams'] = sampling_result["beams"]

        return {category: sequence_all}


def atomic_generate(args, split):
    in_path = os.path.abspath(os.path.join(args.root,
                                           split + '.preprocessed.csv'))
    out_path = in_path.replace('.csv', '.kg2.csv')
    df = pd.read_csv(in_path)

    with working_directory('comet'):
        opt, state_dict = interactive.load_model_file(
            'pretrained_models/atomic_pretrained_model.pickle')

        data_loader, text_encoder = interactive.load_data("atomic", opt)

    n_ctx = data_loader.max_event + data_loader.max_effect
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
        original = re.sub(r'<swap2>.+<swap3>', '', row['text'])
        original = original.replace('<swap1>', '')
        original = ' '.join(original.split())

        result = get_atomic_sequence(original, model, sampler,
                                     data_loader, text_encoder, args.category)

        for key in result:
            df.at[i, 'original-' + key] = json.dumps(result[key]['beams'])

        # edited sentence
        edited = re.sub(r'<swap1>.+<swap2>', '', row['text'])
        edited = edited.replace('<swap3>', '')
        edited = ' '.join(edited.split())

        result = get_atomic_sequence(edited, model, sampler,
                                     data_loader, text_encoder, args.category)

        for key in result:
            df.at[i, 'edited-' + key] = json.dumps(result[key]['beams'])

        df.to_csv(out_path, index=None)


def main():
    args = parse_args()

    for split in ['train', 'dev']:
        atomic_generate(args, split)


if __name__ == "__main__":
    main()
