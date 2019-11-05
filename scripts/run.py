#!/usr/bin/env python3

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from humicroedit.datasets.humicroedit import Humicroedit


def main():
    dataset = Humicroedit('data', 'train')


if __name__ == "__main__":
    main()
