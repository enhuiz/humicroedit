# Humicroedit

## Directory

The directory is out of date, it will be updated later.

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

The model name is in the format of `<framework>-<contextual-feature-extractor>-<loss>`. Currently, the possible choices are list as below:

```
<framework>: baseline, bert
<contextual-feature-extractor>: lstm, transformer
<loss>: sce, mse
```

Where sce stands for soft cross entropy loss and mse for mean squared error loss.

- Train:

```
./scripts/train.py --name bert-transformer-sce
```

- Test:

```
./scripts/test.py --name bert-transformer-sce
```

## History

- 2019-11-26
  1. Add soft cross entropy loss.
  2. Add lemmatization using [spacy](https://spacy.io/).
  3. Use COMET model trained on ATOMIC to relate our data to the corresponding object in the knowledge graph, see `data/humicroedit/task-1/*.kg.csv`.
  4. Add BERT pretraining (only the mask) part.


- 2019-11-27
  1. Use BERT jointly training instead of BERT pretraining (beat the baseline for the first time by :) 0.001).


## Planing

- 2019-11-27
  1. Try to incorporate the knowledge graph into the training.
  2. Run experiments to see whether there is improvement.
  3. Maybe start writing.