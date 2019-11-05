# Humicroedit

## Dataset

The dataset is put into the `data/` folder, you need to [download](https://www.cs.rochester.edu/u/nhossain/humicroedit/semeval-2020-task-7-data.zip) it and unzip by yourself.

```bash
$ wget https://www.cs.rochester.edu/u/nhossain/humicroedit/semeval-2020-task-7-data.zip
$ unzip semeval-2020-task-7-data.zip
$ rm semeval-2020-task-7-data.zip
```

## Directory

```plain
.
├── README.md
├── data                                <- data set
│   ├── task-1
│   │   ├── dev.csv
│   │   └── train.csv
│   └── task-2
│       ├── dev.csv
│       └── train.csv
├── humicroedit                         <- major code
│   ├── __init__.py
│   └── networks
├── scripts                             <- helper scripts
│   └── data
│       └── download.sh
└── semeval-2020-task-7-humicroedit     <- official baseline repo
    ├── README.md
    ├── code
    │   ├── baseline_task_1.py
    │   ├── baseline_task_2.py
    │   ├── score_task_1.py
    │   └── score_task_2.py
    └── output_dev_baseline
        ├── task-1-output.csv
        └── task-2-output.csv
```