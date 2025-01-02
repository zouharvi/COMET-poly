import argparse
import json
import collections
import itertools
import random
import csv
import numpy as np

import tqdm

args = argparse.ArgumentParser()
args.add_argument("mode", choices={"train", "test"})
args.add_argument("-t", "--threshold", default=0.0, type=float)
args = args.parse_args()

if args.mode == "test":
    f_data_in = "data/jsonl/test.jsonl"
elif args.mode == "train":
    f_data_in = "data/jsonl/train.jsonl"

with open(f_data_in) as f:
    data = [json.loads(line) for line in f]

# match based on the source
src_to_tgts = collections.defaultdict(lambda: collections.defaultdict(list))
for x in data:
    src_to_tgts[x["src"]][x["tgt"]].append(x["score"])

src_to_tgts = {
    src:
    [(tgt, np.average(scores)) for tgt, scores in tgts.items()]
    for src, tgts in src_to_tgts.items()
}
src_to_tgts = list(src_to_tgts.items())

if args.mode == "train":
    i_dev = set(random.Random(0).sample(list(range(len(src_to_tgts))), k=500))
    i_train = set(range(len(src_to_tgts))) - i_dev
    src_to_tgts_dev = [src_to_tgts[i] for i in i_dev]
    src_to_tgts_train = [src_to_tgts[i] for i in i_train]
    recipe = [(src_to_tgts_dev, "data/csv/dev_pw.csv"), (src_to_tgts_train, "data/csv/train_pw.csv")]
elif args.mode == "test":
    recipe = [(src_to_tgts, "data/csv/test_pw.csv")]

for src_to_tgts, f_data_out in recipe:
    data_out = []
    for src, l in tqdm.tqdm(src_to_tgts):
        for (mt1, score1), (mt2, score2) in itertools.combinations(l, 2):
            if abs(score1 - score2) <= args.threshold:
                continue
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


    with open(f_data_out, "w") as f:
        writer = csv.DictWriter(f, fieldnames=["src", "mt1", "mt2", "score"])
        writer.writeheader()
        for line in tqdm.tqdm(data_out):
            writer.writerow(line)

# no dedup
# 14098221
# simple dedup
# 11565738
# remove ties
# 11404868
# threshold 10
# 7845268
# threshold 25
# 4099341
# da
# 756942


"""
python3 experiments/03-generate_pw_data.py train -t 0
python3 experiments/03-generate_pw_data.py test -t 0
"""
