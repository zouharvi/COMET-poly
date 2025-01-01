# %%

import collections
import comet
import csv
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
os.chdir("/home/vilda/comet-ranking")


MODEL = "lightning_logs/version_18089135/checkpoints/epoch=2-step=35484-val_kendall=0.292.ckpt"

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
    {"src": src, "mt": tgt}
    for src, tgts in src_to_tgts
    for tgt, _ in tgts
], batch_size=32).scores


src_to_tgts = [
    (src, [
        (tgt, score, scores_pred_all.pop(0))
        for tgt, score in tgts
    ])
    for src, tgts in src_to_tgts
]

# %%
delta_true = []
accuracy = []
for src, tgts in src_to_tgts:
    score_pred = [pred for tgt, score, pred in tgts]
    score_true = [score for tgt, score, pred in tgts]

    for i in range(len(score_pred)):
        for j in range(len(score_pred)):
            delta_true.append(abs(score_true[i] - score_true[j]))
            accuracy.append((score_pred[i] > score_pred[j]) == (score_true[i] > score_true[j]))

print(np.average(accuracy))
bins = np.linspace(min(delta_true), max(delta_true), 10)
# add accuracy to bins based on delta_true
binned_accuracy = []
for i in range(len(bins) - 1):
    binned_accuracy.append(np.average([
        acc
        for delta, acc in zip(delta_true, accuracy)
        if bins[i] <= delta < bins[i + 1]
    ]))

# average bins between two
bins = (bins[:-1] + bins[1:]) / 2

plt.figure(figsize=(3, 2))
plt.bar(bins, binned_accuracy, width=(bins[1] - bins[0])/2)
plt.ylim(0, 1)
plt.text(
    0.05, 0.95,
    f"Average: {np.average(accuracy):.2%}",
    transform=plt.gca().transAxes,
    va="top",
    fontsize=10
)
plt.ylabel("Accuracy")
plt.xlabel("Score $\\Delta$")
plt.savefig("figures/08da-delta_accuracy.pdf")
plt.show()