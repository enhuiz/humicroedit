import os
import copy
import inspect
import argparse
import contextlib
import yaml
import time
from collections import OrderedDict
from functools import reduce


@contextlib.contextmanager
def working_directory(path):
    prev = os.getcwd()
    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def call(callbacks):
    def caller(*args, **kwargs):
        for cb in callbacks:
            cb(*args, **kwargs)
    return caller
