import os

from .humicroedit import HumicroeditDataset
from .examiner import ExaminerDataset
from .toy import ToyDataset


def get(name, split, use_kg):
    name = name.split(os.path.sep)[0]

    if name == 'humicroedit':
        ds = HumicroeditDataset('data/humicroedit/task-1', split, use_kg)
    elif name == 'examiner':
        ds = ExaminerDataset('data/examiner')
    elif name == 'toy':
        ds = ToyDataset()
    else:
        raise Exception("Unknown dataset: {}, avaliable name: "
                        "humicroedit examiner or toy".format(name))

    print(type(ds).__name__, 'loaded.')

    return ds
