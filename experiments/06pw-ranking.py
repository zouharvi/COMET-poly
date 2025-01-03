# %%
import collections
import copy
import tqdm
import csv
import numpy as np
import scipy.stats
import random
import pickle
import json
import os
os.chdir("/home/vilda/comet-ranking")

scores_pred_all = pickle.load(open("data/pickle/scores_pred_pw.pkl", "rb"))

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
ranks_first = []
ranks_last = []
for src, tgts in tqdm.tqdm(src_to_tgts):
    scores_pred = [
        [
            scores_pred_all.pop(0)
            for _ in tgts
        ]
        for _ in tgts
    ]
    # instead of using 0.5, use the median so half are wins and half are losses
    scores_pred = 1.0*(np.array(scores_pred) > np.median(scores_pred))
    # scores_pred = 1.0*(np.array(scores_pred) > 0.5)
    wins = np.sum(scores_pred, axis=1)
    last_rank = len(wins) - np.argmax(wins[::-1]) - 1
    ranks_first.append(np.argmax(wins))
    ranks_last.append(last_rank)
assert len(scores_pred_all) == 0


print(
    f"Average rank (first): {np.average(ranks_first):.2f}",
    f"with averarge number of hypotheses: {np.average([len(tgts) for _, tgts in src_to_tgts]):.2f}",
    f"so percentile {1-np.average(ranks_first)/np.average([len(tgts) for _, tgts in src_to_tgts]):.2%}",
)
print(
    f"Average rank (last):  {np.average(ranks_last):.2f}",
    f"with averarge number of hypotheses: {np.average([len(tgts) for _, tgts in src_to_tgts]):.2f}",
    f"so percentile {1-np.average(ranks_last)/np.average([len(tgts) for _, tgts in src_to_tgts]):.2%}",
)
# Average rank (first): 5.87 with averarge number of hypotheses: 16.13 so percentile 63.62%
# Average rank (last):  8.12 with averarge number of hypotheses: 16.13 so percentile 49.67%