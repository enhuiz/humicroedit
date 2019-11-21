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
├── official                            <- official baseline repo
├── README.md
└── scripts                             <- helper scripts
    ├── data
    │   └── download.sh
    ├── test.py                         <- predict, result will be written to results/
    └── train.py                        <- all training calls this script
```

## Dependencies installation

```
pip3 install -r requirements.txt
```


## Train & test

After downloading the dataset, run:

- Train:

```
./scripts/train.py --name transformer-baseline
```

- Test:

```
./scripts/test.py --name transformer-baseline
```

## NER, EL and KGE

To incorporate the knowledge graph into our task, we may want to understand the following concept:

- Name entity recognition (NER): https://towardsdatascience.com/named-entity-recognition-with-nltk-and-spacy-8c4a7d88e7da
- Entity linking (EL): https://medium.com/analytics-vidhya/entity-linking-a-primary-nlp-task-for-information-extraction-22f9d4b90aa8
- Knowledge graph embeddings (KGE): https://github.com/mnick/scikit-kge