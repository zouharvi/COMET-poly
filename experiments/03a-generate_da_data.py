import argparse
import json
import csv

args = argparse.ArgumentParser()
args.add_argument("data_in")
args.add_argument("data_out")
args = args.parse_args()

with open(args.data_in) as f:
    data = [json.loads(line) for line in f]

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


"""
python3 experiments/04-generate_da_data.py data/jsonl/train.jsonl data/csv/train_da.csv
# take head 1% of data/csv/train_da.csv and save it as data/csv/dev_da.csv
COUNT=500
# clip the first ${COUNT} lines from data/csv/train_da.csv and save it as data/csv/train_da.csv
head -n 500 data/csv/train_da.csv > data/csv/dev_da.csv
head -n 1 data/csv/dev_da.csv > tmp
tail -n +500 data/csv/train_da.csv >> tmp
mv tmp data/csv/train_da.csv
python3 experiments/04-generate_da_data.py data/jsonl/test.jsonl data/csv/test_da.csv
"""