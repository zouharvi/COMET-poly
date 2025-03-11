import argparse
import json
import collections
import random
import csv
import numpy as np
import tqdm

args = argparse.ArgumentParser()
args.add_argument("mode", choices={"train", "test"})
args = args.parse_args()

if args.mode == "test":
    f_data_in = "data/jsonl/test.jsonl"
elif args.mode == "train":
    f_data_in = "data/jsonl/train.jsonl"

with open(f_data_in) as f:
    data = [json.loads(line) for line in f]

# match based on the source and language
src_to_tgts = collections.defaultdict(lambda: collections.defaultdict(list))
for x in data:
    src_to_tgts[(x["src"], x["langs"])][x["tgt"]].append(x["score"])

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
    recipe = [(src_to_tgts_dev, f"data/csv/dev_anchor.csv"), (src_to_tgts_train, f"data/csv/train_anchor.csv")]
elif args.mode == "test":
    recipe = [(src_to_tgts, f"data/csv/test_anchor.csv")]

for src_to_tgts, f_data_out in recipe:
    data_out = []
    for (src, langs), tgts in tqdm.tqdm(src_to_tgts):
        for (mt1, score1) in tgts:
            # randomly select anchor but make sure it's not the same as the current translation
            tgts_valid = [(tgt, score) for tgt, score in tgts if tgt != mt1]
            if len(tgts_valid) == 0:
                mt2 = ""
                score2 = 0
            else:
                mt2, score2 = random.choice(tgts)
            data_out.append({
                "src": src,
                "mt": mt1,
                "score": score1,
                "mt2": mt2,
                "score2": score2,
            })

    with open(f_data_out, "w") as f:
        writer = csv.DictWriter(f, fieldnames=["src", "mt", "score", "mt2", "score2"])
        writer.writeheader()
        for line in tqdm.tqdm(data_out):
            writer.writerow(line)

"""
python3 experiments/03c-generate_anchor_data.py train
python3 experiments/03c-generate_anchor_data.py test
"""
