import collections
import copy

import tqdm
import comet
import csv
import numpy as np
import argparse
import scipy.stats
import random

args = argparse.ArgumentParser()
args.add_argument("model")
args = args.parse_args()


model = comet.load_from_checkpoint(args.model)

# # evaluate pairwise comparison
# # TODO: change to 10_000
# data = random.Random(0).sample(list(csv.DictReader(open("data/csv/test_pairwise.csv"))), k=1_000)
# scores_pred = model.predict(data, batch_size=32).scores
# acc = np.average([
#     (pred > 0.5)*1.0 == float(line["score"])
#     for pred, line in zip(scores_pred, data)
# ])
# print(f"Accuracy (pairwise ranking task): {acc:.4f}")


# evaluate top-1 ranking
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

ranks = []
pw_accuracy = []
for src, tgts in tqdm.tqdm(src_to_tgts):
    scores_pred = [
        [
            scores_pred_all.pop(0)
            for _ in tgts
        ]
        for _ in tgts
    ]
    # instead of using 0.5, use the median so half are wins and half are losses
    # scores_pred = 1.0*(np.array(scores_pred) > np.median(scores_pred))
    scores_pred = 1.0*(np.array(scores_pred) > 0.5)
    wins = np.sum(scores_pred, axis=1)
    ranks.append(np.argmax(wins))


    scores_true = [
        [
            float(score1 > score2)
            for tgt2, score2 in tgts
        ]
        for tgt1, score1 in tgts
    ]
    scores_pred = np.array(scores_pred)
    scores_true = np.array(scores_true)
    pw_accuracy += (scores_pred == scores_true).flatten().tolist()

    # print(scores_true)
    # print(scores_pred)
    # print(scores_pred == scores_true)
    # print(ranks)


print(f"Pairwise accuracy: {np.average(pw_accuracy):.2%}")
print(
    f"Average rank: {np.average(ranks):.2f}",
    f"with averarge number of hypotheses: {np.average([len(tgts) for _, tgts in src_to_tgts]):.2f}",
    f"so percentile {1-np.average(ranks)/np.average([len(tgts) for _, tgts in src_to_tgts]):.2%}",
)
# Average rank: 5.81 with averarge number of hypotheses: 20.16 so percentile 71.20%