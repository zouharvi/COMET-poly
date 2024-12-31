import argparse
import json
import collections
import itertools
import random
import csv

import tqdm

args = argparse.ArgumentParser()
args.add_argument("data_in")
args.add_argument("data_out")
args.add_argument("-s", "--subsample", default=1, type=float)
args = args.parse_args()

with open(args.data_in) as f:
    data = [json.loads(line) for line in f]

# match based on the source
src_to_tgts = collections.defaultdict(list)
for x in data:
    src_to_tgts[x["src"]].append((x["tgt"], x["score"]))


data_out = []
for src, l in tqdm.tqdm(list(src_to_tgts.items())):
    combs_all = list(itertools.combinations(l, 2))
    combs_all_uniq = set(x for l in combs_all for x in l)
    combs = []
    # prune some pairs but make sure all systems are present
    while len(combs_all_uniq) > len(set(x for l in combs for x in l)):
        combs = [x for x in combs_all if random.random() <= args.subsample]

    for (mt1, score1), (mt2, score2) in combs:
        # randomly flip coin which one is better
        tmp = [(mt1, score1), (mt2, score2)]
        random.shuffle(tmp)
        (mt1, score1), (mt2, score2) = tmp
        data_out.append({
            "src": src,
            "mt1": mt1,
            "mt2": mt2,
            "score": (score1 > score2)*1,
        })


with open(args.data_out, "w") as f:
    writer = csv.DictWriter(f, fieldnames=["src", "mt1", "mt2", "score"])
    writer.writeheader()
    for line in tqdm.tqdm(data_out):
        writer.writerow(line)

# python3 experiments/03-generate_pairwise_data.py data/jsonl/train.jsonl data/csv/train_pairwise.csv -s 0.1

# compute 1% of train_pairwise.csv and store it as COUNT
# COUNT=$(wc -l < data/csv/train_pairwise.csv)
# COUNT=$((COUNT*1/100))
# take head ${COUNT} of data/csv/train_pairwise.csv and save it as data/csv/dev_pairwise.csv
# clip the first ${COUNT} lines from data/csv/train_pairwise.csv and save it as data/csv/train_pairwise.csv
# head -n ${COUNT} data/csv/train_pairwise.csv > data/csv/dev_pairwise.csv
# head -n 1 data/csv/dev_pairwise.csv > tmp
# tail -n +${COUNT} data/csv/train_pairwise.csv >> tmp
# mv tmp data/csv/train_pairwise.csv
# python3 experiments/03-generate_pairwise_data.py data/jsonl/test.jsonl data/csv/test_pairwise.csv
