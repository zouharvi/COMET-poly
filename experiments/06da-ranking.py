# %%
import collections
import csv
import numpy as np
import random
import pickle
import json
import os
os.chdir("/home/vilda/comet-ranking")

scores_pred_all = pickle.load(open("data/pickle/scores_pred_da.pkl", "rb"))

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

ranks_last = []
ranks_first = []
for src, tgts in src_to_tgts:
    scores_pred = [
        scores_pred_all.pop(0)
        for _ in tgts
    ]
    rank_first = np.argmax(scores_pred)
    rank_last = len(scores_pred) - np.argmax(scores_pred[::-1]) - 1
    ranks_first.append(rank_first)
    ranks_last.append(rank_last)
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
# Average rank (first): 6.61 with averarge number of hypotheses: 16.13 so percentile 59.00%
# Average rank (last):  7.32 with averarge number of hypotheses: 16.13 so percentile 54.61%