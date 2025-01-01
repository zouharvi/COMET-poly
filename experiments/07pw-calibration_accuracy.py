# %%

import collections
import comet
import csv
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
os.chdir("/home/vilda/comet-ranking")


MODEL = "lightning_logs/version_18089134/checkpoints/epoch=0-step=24307-val_accuracy=0.650.ckpt"

# args = argparse.ArgumentParser()
# args.add_argument("model")
# args = args.parse_args()

model = comet.load_from_checkpoint(MODEL)


# load data
data = list(csv.DictReader(open("data/csv/test_da.csv")))
src_to_tgts = collections.defaultdict(list)
for x in data:
    src_to_tgts[x["src"]].append((x["mt"], float(x["score"])))
src_to_tgts = {
    src: sorted(tgts, key=lambda x: x[1], reverse=True)
    for src, tgts in src_to_tgts.items()
    # take sources with at least 10 translations
    if len(tgts) >= 10
}
src_to_tgts = list(src_to_tgts.items())[:100]


scores_pred_all = model.predict([
    {"src": src, "mt1": tgt1, "mt2": tgt2}
    for src, tgts in src_to_tgts
    for tgt1, _ in tgts
    for tgt2, _ in tgts
], batch_size=32).scores

src_to_tgts = [
    (src, [
        (tgt1, tgt2, score1, score2, scores_pred_all.pop(0))
        for tgt1, score1 in tgts
        for tgt2, score2 in tgts
    ])
    for src, tgts in src_to_tgts
]

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

plt.bar(bins, binned_accuracy, width=(bins[1] - bins[0])/2)
plt.plot(plt.xlim(), [0.5, 1], color="red")
plt.ylim(0, 1)