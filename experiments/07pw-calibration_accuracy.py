# %%

import collections
import comet_multi_cand
import csv
import numpy as np
import argparse
import pickle
import json
import matplotlib.pyplot as plt
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

src_to_tgts = [
    (src, [
        (tgt1, tgt2, score1, score2, scores_pred_all.pop(0))
        for tgt1, score1 in tgts
        for tgt2, score2 in tgts
    ])
    for src, tgts in src_to_tgts
]
assert len(scores_pred_all) == 0

# %%
delta_pred = []
accuracy = []
for src, tgts in src_to_tgts:
    delta_pred += [abs(pred-0.5) for (tgt1, tgt2, score1, score2, pred) in tgts]
    accuracy += [(pred > 0.5)*1.0 == float(score1 > score2) for (tgt1, tgt2, score1, score2, pred) in tgts]
    

bins = np.linspace(min(delta_pred), max(delta_pred), 10)
# add accuracy to bins based on delta_true
binned_accuracy = []
for i in range(len(bins) - 1):
    binned_accuracy.append(np.average([
        acc
        for delta, acc in zip(delta_pred, accuracy)
        if bins[i] <= delta < bins[i + 1]
    ]))

# average bins between two
bins = (bins[:-1] + bins[1:]) / 2

plt.figure(figsize=(2.5, 1.5))
plt.bar(bins, binned_accuracy, width=(bins[1] - bins[0])/2)
plt.plot(plt.xlim(), [0.5, 1], color="red")
plt.ylim(0, 1)
plt.ylabel("Accuracy", labelpad=-5)
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
plt.xlabel("Confidence", labelpad=-10)
plt.xticks(
    [min(bins), max(bins)],
    ["0%", "100%"],
)
plt.tight_layout(pad=0)
plt.savefig("figures/07pw-calibration_accuracy.pdf")
plt.show()