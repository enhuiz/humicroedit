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

## NER, EL and KGE

To incorporate the knowledge graph into our task, we may want to understand the following concept:

NER (name entity recognition): https://towardsdatascience.com/named-entity-recognition-with-nltk-and-spacy-8c4a7d88e7da
EL (entity linking): https://medium.com/analytics-vidhya/entity-linking-a-primary-nlp-task-for-information-extraction-22f9d4b90aa8
KGE (knowledge graph embeddings): https://github.com/mnick/scikit-kge