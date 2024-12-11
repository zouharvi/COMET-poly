import argparse
import json
import collections
import itertools
import random
import csv

args = argparse.ArgumentParser()
args.add_argument("data_in")
args.add_argument("data_out")
args.add_argument("-c", "--crop", default=1, type=float)
args = args.parse_args()

with open(args.data_in) as f:
    data = [json.loads(line) for line in f]

data = data[:int(len(data)*args.crop)]

data_out = []
for line in data:
    data_out.append({
        "src": line["src"],
        "mt": line["tgt"],
        "score": line["score"],
    })


with open(args.data_out, "w") as f:
    writer = csv.DictWriter(f, fieldnames=["src", "mt", "score"])
    writer.writeheader()
    writer.writerows(data_out)


# python3 experiments/04-generate_da_data.py data/jsonl/train.jsonl data/csv/train_da.csv -c 0.1
# python3 experiments/04-generate_da_data.py data/jsonl/test.jsonl data/csv/test_da.csv
# python3 experiments/04-generate_da_data.py data/jsonl/dev.jsonl data/csv/dev_da.csv
