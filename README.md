# COMET-poly

This repository contains a version of COMET which supports multiple translations to be scored.
Further instructions WIP.

Importantly, this repository hosts the `comet-poly` package, which is a fork of [`unbabel-comet`](https://github.com/Unbabel/COMET/) that is not compatible.
Thus, you need to install this package by:
```
pip install "git+https://github.com/zouharvi/COMET-poly#egg=comet-poly&subdirectory=comet_poly"
```
or alternatively by cloning and installing locally:
```
git clone https://github.com/zouharvi/COMET-poly
cd COMET-poly
pip install -e comet_poly
```

## Running pre-trained models

TODO

## Training models and replicating experiments

The scripts for training new COMET models and running the experiments in the paper are in `experiments/`.
First, start by fetching the data:
```
python3 experiments/01-get_data_same.py
python3 experiments/01-get_data_same.py --sort-by-sim # optional, takes a long time
python3 experiments/01-get_data_retrieval.py # see arguments
```

TODO