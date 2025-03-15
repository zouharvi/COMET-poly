import comet_multi_cand
import argparse
import csv
import numpy as np

args = argparse.ArgumentParser()
args.add_argument("--model", default="TODO")
args = args.parse_args()

model = comet_multi_cand.load_from_checkpoint(args.model)
data = list(csv.DictReader(open("data/csv/test_multi.csv")))
scores_pred = model.predict(data, batch_size=32).scores

accuracy = np.average([
    (float(x["score"])>float(x["score2"])) == (y_pred[0]>y_pred[1])
    for x, y_pred in zip(data, scores_pred)
])
print(f"Accuracy: {accuracy:.3f}")

# TODO: eval on each language separatedly
