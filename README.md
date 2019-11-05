# Humicroedit

## Dataset

The dataset is put into the `data/` folder, you need to [download](https://www.cs.rochester.edu/u/nhossain/humicroedit/semeval-2020-task-7-data.zip) it and unzip by yourself or run `scripts/data/download.sh`.

```bash
$ wget https://www.cs.rochester.edu/u/nhossain/humicroedit/semeval-2020-task-7-data.zip
$ unzip semeval-2020-task-7-data.zip
$ rm semeval-2020-task-7-data.zip
```

## Directory

```plain
.
├── data                                <- data set
│   ├── task-1
│   │   ├── dev.csv
│   │   └── train.csv
│   └── task-2
│       ├── dev.csv
│       └── train.csv
├── humicroedit                         <- major code
│   ├── datasets
│   │   ├── humicroedit.py              <- pytorch loader for dataset, texts get preprocessed here. 
│   │   ├── __init__.py
│   │   └── vocab.py                    <- vocab, which records the index of each word, indices are used for training model instead of word.
│   ├── __init__.py
│   └── networks
│       ├── decoders
│       │   └── __init__.py
│       ├── encoders
│       │   └── __init__.py
│       ├── __init__.py
│       └── layers.py
├── README.md
├── scripts                             <- helper scripts
│   ├── data
│   │   └── download.sh
│   └── run.py
└── semeval-2020-task-7-humicroedit     <- official baseline repo

