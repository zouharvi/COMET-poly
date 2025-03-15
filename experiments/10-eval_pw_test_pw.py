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
scores_pred = model.predict(data, batch_size=32).scores

accuracy = np.average([
    (x["langs"], (float(x["score"])>float(x["score2"])) == (y_pred[0]>y_pred[1]))
    for x, y_pred in zip(data, scores_pred)
])

# average per language first and then overall (macro-average)
accuracy_lang = collections.defaultdict(list)
for lang, acc in accuracy:
    accuracy_lang[lang].append(acc)
accuracy_avg = np.average([np.average(acc) for acc in accuracy_lang.values()])
print(f"Accuracy: {accuracy_avg:.3f}")