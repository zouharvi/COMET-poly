import comet_multi_cand
import argparse
import csv
import numpy as np
import collections

args = argparse.ArgumentParser()
args.add_argument("model")
args = args.parse_args()

model = comet_multi_cand.load_from_checkpoint(args.model)
data = list(csv.DictReader(open("data/csv/test_multi.csv")))
# take only the data where the difference is at least 10
data = [
    x for x in data
    if abs(float(x["score"]) - float(x["score2"])) >= 10
]
scores_pred = model.predict(data, batch_size=64).scores

accuracy = [
    (x["langs"], (float(x["score"])>float(x["score2"])) == (y_pred[0]>y_pred[1]))
    for x, y_pred in zip(data, scores_pred)
]

# average per language first and then overall (macro-average)
accuracy_lang = collections.defaultdict(list)
for lang, acc in accuracy:
    accuracy_lang[lang].append(acc)

# print for each lang
for lang, acc in accuracy_lang.items():
    print(f"{lang:>5} {np.average(acc):.3f}")
accuracy_avg = np.average([np.average(acc) for acc in accuracy_lang.values()])
print(f"Accuracy: {accuracy_avg:.3f}")