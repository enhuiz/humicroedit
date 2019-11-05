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
├── README.md
├── data                                <- dataset, readonly.
│   ├── task-1
│   │   ├── dev.csv
│   │   └── train.csv
│   └── task-2
│       ├── dev.csv
│       └── train.csv
├── humicroedit                         <- major code
│   ├── __init__.py
│   ├── datasets
│   │   ├── __init__.py
│   │   ├── humicroedit.py              <- pytorch loader for the dataset, texts get preprocessed here. 
│   │   └── vocab.py                    <- vocab, which records the index of each word, indices are used for training model instead 
│   ├── networks
│   │   ├── __init__.py                 <- all models are designed here
│   │   ├── encoders
│   │   │   ├── __init__.py
│   │   │   └── lstm.py                 <- a residue LSTM encoder
│   │   ├── layers.py
│   │   └── losses
│   │       ├── __init__.py
│   │       └── mean_squared_error.py   <- mse loss for regression
│   └── utils.py
├── scripts                             <- helper scripts
│   ├── data
│   │   └── download.sh
│   ├── test.py                         <- predict, result will be written to results/
│   └── train.py                        <- all training calls this script
└── semeval-2020-task-7-humicroedit     <- official baseline repo
```