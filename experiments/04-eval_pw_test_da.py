import comet_multi_cand
import argparse
import csv
import numpy as np

args = argparse.ArgumentParser()
args.add_argument("--model", default="TODO")
args = args.parse_args()

model = comet_multi_cand.load_from_checkpoint(args.model)
data = list(csv.DictReader(open("data/csv/test_multi.csv")))
scores1_pred = model.predict(data, batch_size=32).scores
for line in data:
    line["mt"] = line["mt2"]
scores2_pred = model.predict(data, batch_size=32).scores

accuracy = np.average([
    (float(x["score"])>float(x["score2"])) == (y1_pred>y2_pred)
    for x, y1_pred, y2_pred in zip(data, scores1_pred, scores2_pred)
])
print(f"Accuracy: {accuracy:.3f}")