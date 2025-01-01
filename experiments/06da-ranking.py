import collections
import comet
import csv
import numpy as np
import argparse
import random

args = argparse.ArgumentParser()
args.add_argument("model")
args = args.parse_args()

model = comet.load_from_checkpoint(args.model)

# evaluate top-1 ranking and pairwise accuracy
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

ranks = []
for src, tgts in src_to_tgts:
    scores_pred = [
        scores_pred_all.pop(0)
        for _ in tgts
    ]
    ranks.append(np.argmax(scores_pred))

print(
    f"Average rank: {np.average(ranks):.2f}",
    f"with averarge number of hypotheses: {np.average([len(tgts) for _, tgts in src_to_tgts]):.2f}",
    f"so percentile {1-np.average(ranks)/np.average([len(tgts) for _, tgts in src_to_tgts]):.2%}",
)
# Average rank: 8.09 with averarge number of hypotheses: 20.16 so percentile 59.88%