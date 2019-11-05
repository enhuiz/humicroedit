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
    parser.add_argument('--name', type=str, default='baseline')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--root', type=str, default='data/task-1')
    parser.add_argument('--device', type=str, default='cuda')
    opts = parser.parse_args()
    return opts


def test(model, dataloader, optimizer, epochs,
         on_iteration_start=[],
         on_iteration_end=[]):

    status = type('Status', (), {
        'model': model,
    })

    status.pbar = tqdm.tqdm(dataloader)

    for batch in status.pbar:
        status.batch = batch
        call(on_iteration_start)(status)

        out = model(batch)

        status.out = out
        call(on_iteration_end)(status)


def main():
    opts = get_opts()

    os.makedirs(os.path.join('results', opts.name), exist_ok=True)
    results = []

    def save_results(status):
        results += list(zip(status.out['id'], status.out['x']))
        df = pd.DataFrame(results, columns=['id', 'pred'])
        df.to_csv(os.path.join('results', opts.name, 'pred.csv'))

    # build dataset
    ds = HumicroeditDataset(opts.root, 'dev')
    dataloader = DataLoader(ds,
                            batch_size=opts.batch_size,
                            shuffle=True,
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

    # build optimizer
    optimizer = torch.optim.Adam(model.parameters(), opts.lr)

    test(model,
         dataloader,
         optimizer,
         opts.epochs,
         on_iteration_end=[
             save_results,
         ])


if __name__ == "__main__":
    main()
