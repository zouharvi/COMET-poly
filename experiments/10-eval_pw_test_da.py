import comet_multi_cand
import argparse
import csv
import numpy as np
import collections

args = argparse.ArgumentParser()
args.add_argument("--model", default="TODO")
args = args.parse_args()

model = comet_multi_cand.load_from_checkpoint(args.model)
data = list(csv.DictReader(open("data/csv/test_multi.csv")))
scores1_pred = model.predict(data, batch_size=32).scores
for line in data:
    line["mt"] = line["mt2"]
scores2_pred = model.predict(data, batch_size=32).scores

accuracy = [
    (x["langs"], (float(x["score"]) > float(x["score2"])) == (y1_pred > y2_pred))
    for x, y1_pred, y2_pred in zip(data, scores1_pred, scores2_pred)
]
# average per language first and then overall (macro-average)
accuracy_lang = collections.defaultdict(list)
for lang, acc in accuracy:
    accuracy_lang[lang].append(acc)
accuracy_avg = np.average([np.average(acc) for acc in accuracy_lang.values()])
print(f"Accuracy: {accuracy_avg:.3f}")