#!/usr/bin/env python3
import os
import sys

import glob
import time
import tqdm
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from test import test

try:
    sys.path.append('.')
    from humicroedit import datasets
    from humicroedit import networks
    from humicroedit.utils import call
    from humicroedit.datasets.vocab import vocab
except ImportError as e:
    print(e)
    print('Please run under the root dir, but not {}.'.format(os.getcwd()))
    exit()


def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str,
                        default='baseline-lstm-mse@humicroedit')
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--vocab', type=str, default='data/examiner/'
                        'examiner-date-text.preprocessed.csv')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save-every', type=int, default=1)
    opts = parser.parse_args()
    opts.name = opts.name.lower()
    opts.name = opts.name.replace('/', os.path.sep, 1)
    return opts


def log(status):
    epoch = status.epoch
    loss = np.mean(status.losses, axis=0)
    if loss.ndim == 0:
        loss = loss[None]
    with np.printoptions(precision=4, suppress=True):
        msg = 'Epoch {} loss {}'.format(epoch, loss)
    status.pbar.set_description(msg)


def train(model, dataloader, optimizer, epochs,
          on_iteration_start=[],
          on_iteration_end=[],
          on_epoch_start=[],
          on_epoch_end=[],
          pbar=tqdm.tqdm):
    model = model.train()

    on_iteration_end += [log]
    on_epoch_start += [lambda status: status.losses.clear()]

    status = type('Status', (), {
        'epoch': 1,
        'model': model,
        'losses': [],
    })

    while status.epoch <= epochs:
        call(on_epoch_start)(status)

        if status.epoch > epochs:
            break

        status.pbar = pbar(dataloader)

        for batch in status.pbar:
            try:
                status.batch = batch
                call(on_iteration_start)(status)

                out = model(feed=batch)
                out['loss'].sum().backward()
                optimizer.step()
                optimizer.zero_grad()

                status.out = out
                status.losses.append(out['loss'].detach().cpu().numpy())
                call(on_iteration_end)(status)
            except RuntimeError as e:
                print(e)
                model.zero_grad()
                optimizer.zero_grad()
                torch.cuda.empty_cache()

        call(on_epoch_end)(status)

        status.epoch += 1


def main():
    opts = get_opts()

    vocab.load(opts.vocab)

    ckpts = sorted(glob.glob(os.path.join('ckpt', opts.name, '*.pth')))

    # build dataset
    ds = datasets.get(opts.name, 'train', 'kg' in opts.name)

    num_train = int(len(ds) * 0.9)

    train_ds = Subset(ds, range(num_train))
    valid_ds = Subset(ds, range(num_train, len(ds)))

    train_dl = DataLoader(train_ds,
                          batch_size=opts.batch_size,
                          shuffle=True,
                          collate_fn=ds.get_collate_fn())

    valid_dl = DataLoader(valid_ds,
                          batch_size=opts.batch_size,
                          shuffle=True,
                          collate_fn=ds.get_collate_fn())

    # build callbacks
    def load_model(status):
        if not ckpts:
            return
        ckpt = ckpts[-1]
        epoch = int(Path(ckpt).stem)
        if epoch == opts.epochs:
            status.epoch = epoch + 1
            print('{} already exists.'.format(ckpt))
        elif status.epoch <= epoch:
            status.model.load_state_dict(torch.load(ckpt, map_location='cpu'))
            status.model.to(opts.device)
            status.epoch = epoch + 1
            print(ckpt, 'loaded.')

    def save_model(status):
        model = status.model
        i = status.epoch
        if i % opts.save_every == 0:
            path = os.path.join('ckpt', opts.name, '{0:05d}.pth'.format(i))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(model.state_dict(), path)

    def validate(status):
        losses = []
        test(status.model,
             valid_dl,
             on_iteration_end=[
                 lambda status: losses.append(
                     status.out['loss'].detach().cpu().numpy()
                 )
             ],
             pbar=lambda dl: dl)
        status.model.train()
        loss = np.mean(losses, axis=0)
        if loss.ndim == 0:
            loss = loss[None]
        with np.printoptions(precision=4, suppress=True):
            print('Loss on valid set: {}'.format(loss))

    # build model
    model = networks.get(opts.name).to(opts.device)

    # build optimizer
    optimizer = torch.optim.Adam(model.parameters(), opts.lr)

    train(model,
          train_dl,
          optimizer,
          opts.epochs,
          on_epoch_start=[
              load_model,
          ],
          on_epoch_end=[
              save_model,
              validate,
          ])


if __name__ == "__main__":
    main()
