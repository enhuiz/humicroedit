#!/usr/bin/env python3
import os
import sys

import glob
import time
import tqdm
import argparse
from pathlib import Path

import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


try:
    sys.path.append('.')
    from humicroedit.datasets.humicroedit import HumicroeditDataset
    from humicroedit import networks
    from humicroedit.utils import call
except Exception as e:
    print(e)
    print('Please run inside the root dir, but not {}.'.format(os.getcwd()))
    exit()


def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='baseline-lstm-mse')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--root', type=str, default='data/humicroedit/task-1')
    parser.add_argument('--device', type=str, default='cuda')
    opts = parser.parse_args()
    opts.name = opts.name.lower()
    return opts


def test(model, dl,
         on_iteration_start=[],
         on_iteration_end=[],
         pbar=tqdm.tqdm):
    model = model.eval()

    status = type('Status', (), {
        'model': model,
        'pbar': pbar(dl),
    })

    for batch in status.pbar:
        status.batch = batch
        call(on_iteration_start)(status)

        out = model(batch)

        status.out = out
        call(on_iteration_end)(status)


def main():
    opts = get_opts()

    results = []

    def save_results(status):
        nonlocal results
        ids = status.out['id']
        preds = status.out['pred'].tolist()
        results += list(zip(ids, preds))
        df = pd.DataFrame(results, columns=['id', 'pred'])
        # force the pred inside its domain
        df['pred'] = df['pred'].clip(0, 3)
        path = os.path.join('results', opts.name, 'task-1-output.csv')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=None)

    # build dataset
    ds = HumicroeditDataset(opts.root, 'dev', use_kg='kg' in opts.name)
    dl = DataLoader(ds,
                    batch_size=opts.batch_size,
                    shuffle=False,
                    collate_fn=ds.get_collate_fn())

    try:
        ckpt = sorted(glob.glob(os.path.join('ckpt', opts.name, '*.pth')))[-1]
    except:
        raise Exception('ERROR: No ckpt found for "{}".'.format(opts.name))

    # load model
    model = networks.get(opts.name)
    model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    model = model.to(opts.device)
    print('{} loaded.'.format(ckpt))

    test(model,
         dl,
         on_iteration_end=[
             save_results,
         ])


if __name__ == "__main__":
    main()
