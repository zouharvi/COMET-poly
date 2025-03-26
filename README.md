# COMET-multi-cand

This repository contains a version of COMET which supports multiple translations to be scored.
Further instructions WIP.

Importantly, this repository hosts the `comet-multi-cand` package, which is a fork of [`unbabel-comet`](https://github.com/Unbabel/COMET/) that is not compatible.
Thus, you need to install this package by:
```
pip install "git+https://github.com/zouharvi/COMET-multi-cand#egg=comet-multi-cand&subdirectory=comet_multi_cand"
```
or alternatively by cloning and installing locally:
```
git clone https://github.com/zouharvi/COMET-multi-cand
cd COMET-multi-cand
pip install -e comet_multi_cand
```

## Running pre-trained models

TODO

## Training models and replicating experiments

The scripts for training new COMET models and running the experiments in the paper are in `experiments/`.
First, start by fetching the data:
```
python3 experiments/01-get_data_multi.py
python3 experiments/01-get_data_multi.py --sort-by-sim # (optional, takes a long time)
```

TODO