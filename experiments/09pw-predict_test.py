import collections
import copy
import tqdm
import comet
import csv
import numpy as np
import argparse
import scipy.stats
import json
import random

args = argparse.ArgumentParser()
args.add_argument("--model", default="lightning_logs/version_19759459/checkpoints/epoch=0-step=8750-val_accuracy=0.586.ckpt")
args = args.parse_args()

model = comet.load_from_checkpoint(args.model)

# load data
data = [json.loads(x) for x in open("data/jsonl/test.jsonl")]
src_to_tgts = collections.defaultdict(list)
for x in data:
    src_to_tgts[(x["src"], x["langs"])].append((x["tgt"], x["score"]))
src_to_tgts = {
    src: sorted(tgts, key=lambda x: x[1], reverse=True)
    for src, tgts in src_to_tgts.items()
    # take sources with at least 2 translations
    if len(tgts) >= 2
}
src_to_tgts = list(src_to_tgts.items())[:1_000]

scores_pred_all = model.predict([
    {"src": src, "mt1": tgt1, "mt2": tgt2}
    for (src, langs), tgts in src_to_tgts
    for tgt1, _ in tgts
    for tgt2, _ in tgts
], batch_size=32).scores

pickle.dump(scores_pred_all, open("data/pickle/scores_pred_pw.pkl", "wb"))