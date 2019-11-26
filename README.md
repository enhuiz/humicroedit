# Humicroedit

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

## Setup

### Clone the project

```
git clone --recursive https://github.com/enhuiz/humicroedit 
```

### Install dependencies & download dataset

```
./scripts/setup/humicroedit.sh
./scripts/setup/comet.sh
```

Some datasets are provided in the repo so there is no need to download.

### Download pretrained models for COMET

Please manually download the pretrained model from [here](https://drive.google.com/open?id=1FccEsYPUHnjzmX-Y5vjCBeyRt1pLo8FB) and untar it into `comet/pretrained_model`.

## Preprocess

Apply basic preprocess on the sentence:

```
./data/preprocess.py
```

Fetch COMET object given the subject over all relations (this step is not necessary):

```
./data/comet.py
```

## Train & test

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