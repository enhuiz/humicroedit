#!/usr/bin/env python3

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from humicroedit.datasets.humicroedit import Humicroedit
from torch.utils.data import DataLoader


def main():
    dataset = Humicroedit('data', 'train')
    print(dataset)
    dataloader = DataLoader(dataset, batch_size=8,
                            collate_fn=dataset.get_collate_fn())
    for x, y in dataloader:
        print(x)
        print(y)
        break


if __name__ == "__main__":
    main()
