#!/usr/bin/env python3
import os
import sys

import glob
import time
import tqdm
import argparse

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
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--root', type=str, default='data/task-1')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save-every', type=int, default=1)
    opts = parser.parse_args()
    return opts


def train(model, dataloader, optimizer, epochs,
          on_iteration_start=[],
          on_iteration_end=[],
          on_epoch_start=[],
          on_epoch_end=[]):

    status = type('Status', (), {
        'epoch': 0,
        'model': model,
    })

    while status.epoch < epochs:
        status.pbar = tqdm.tqdm(dataloader)
        call(on_epoch_start)(status)

        for batch in status.pbar:
            status.batch = batch
            call(on_iteration_start)(status)

            out = model({'x': batch[0], 'y': batch[1]})
            out['loss'].backward()
            optimizer.step()
            optimizer.zero_grad()

            status.out = out
            call(on_iteration_end)(status)

        call(on_epoch_end)(status)


def main():
    opts = get_opts()

    # build callbacks
    ckpts = sorted(glob.glob(os.path.join('ckpt', opts.name, '*.pth')))

    def load_model(status):
        if not ckpts:
            return
        ckpt = ckpts[-1]
        epoch = int(ckpt.split('.pth')[0])
        if status.epoch < epoch:
            status.epoch = epoch
            status.model.load_state_dict(torch.load(ckpt, location='cpu'))
            status.model.to(opts.device)

    def save_model(status):
        model = status.model
        i = status.epoch
        if i % opts.save_every == 0:
            path = os.path.join('ckpt', opts.name, '{0:05d}.pth'.format(i))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(model.state_dict(), path)

    def log(status):
        loss = status.out['loss'].item()
        msg = 'loss {:.4g}'.format(loss)
        status.pbar.set_description(msg)

    # build dataset
    ds = HumicroeditDataset(opts.root, 'train')
    dataloader = DataLoader(ds,
                            batch_size=opts.batch_size,
                            shuffle=True,
                            collate_fn=ds.get_collate_fn())

    # build model
    model = networks.get(opts.name).to(opt.device)

    # build optimizer
    optimizer = torch.optim.Adam(model.parameters(), opts.lr)

    train(model,
          dataloader,
          optimizer,
          opts.epochs,
          on_iteration_end=[
              log,
          ],
          on_epoch_start=[
              load_model,
          ],
          on_epoch_end=[
              save_model,
          ])


if __name__ == "__main__":
    main()
