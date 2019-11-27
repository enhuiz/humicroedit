import os

from .humicroedit import HumicroeditDataset
from .examiner import ExaminerDataset


def get(root, split, use_kg):
    if 'humicroedit' in root:
        ds = HumicroeditDataset(root, split, use_kg)
    if 'examiner' in root:
        ds = ExaminerDataset(root, split)
    else:
        raise Exception("Unknown dataset: {}, "
                        "please rename your dataset folder to "
                        "humicroedit or examiner".format(root))
    print(type(ds).__name__, 'loaded.')
    return ds
