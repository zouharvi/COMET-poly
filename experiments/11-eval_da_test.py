import comet_multi_cand
import argparse
import csv
import scipy.stats

args = argparse.ArgumentParser()
args.add_argument("--model", default="TODO")
args = args.parse_args()

model = comet_multi_cand.load_from_checkpoint(args.model)
data = list(csv.DictReader(open("data/csv/test_multi.csv")))
scores_pred = model.predict(data, batch_size=32).scores

corr_pearson = scipy.stats.pearsonr([float(x["score"]) for x in data], scores_pred)[0]
corr_kendal = scipy.stats.kendalltau([float(x["score"]) for x in data], scores_pred, variant="b")[0]
print(f"ρ={corr_pearson:.3f} τ={corr_kendal:.3f}")


# TODO: eval on each language separatedly
