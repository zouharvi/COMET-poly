import comet
import csv
import numpy as np
import argparse
import random

args = argparse.ArgumentParser()
args.add_argument("model")
args = args.parse_args()

data = random.Random(0).sample(list(csv.DictReader(open("data/csv/test_pairwise.csv"))), k=5000)
data_uniq = list({
    (line["src"], line["mt1"])
    for line in data
} | {
    (line["src"], line["mt2"])
    for line in data
})

model = comet.load_from_checkpoint(args.model)
# model = comet.load_from_checkpoint(comet.download_model("Unbabel/wmt22-cometkiwi-da"))

# evaluate pairwise comparison
scores_pred = model.predict([
    {"src": src, "mt": tgt}
    for src, tgt in data_uniq   
]).scores

srctgt_to_score = {}
for (src, tgt), score in zip(data_uniq, scores_pred):
    srctgt_to_score[(src, tgt)] = score

scores_pred = [
    srctgt_to_score[(line["src"], line["mt1"])] > srctgt_to_score[(line["src"], line["mt2"])]
    for line in data
]

acc = np.average(
    (np.array(scores_pred)>0.5)*1.0 == [float(line["score"]) for line in data]
)
print(f"Accuracy (pairwise ranking task): {acc:.4f}")