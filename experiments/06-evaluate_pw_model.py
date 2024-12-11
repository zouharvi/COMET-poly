import comet
import csv
import numpy as np
import argparse
import scipy.stats
import random

args = argparse.ArgumentParser()
args.add_argument("model")
args = args.parse_args()

data = random.Random(0).sample(list(csv.DictReader(open("data/csv/test_pairwise.csv"))), k=5000)

model = comet.load_from_checkpoint(args.model)

# evaluate pairwise comparison
scores_pred = model.predict(data).scores
acc = np.average(
    (np.array(scores_pred)>0.5)*1.0 == [float(line["score"]) for line in data]
)
print(f"Accuracy (pairwise ranking task): {acc:.4f}")