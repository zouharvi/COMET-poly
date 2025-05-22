import comet_multi_cand
import argparse
import csv
import utils

args = argparse.ArgumentParser()
args.add_argument("model")
args.add_argument("--data", default="data/csv/test_same_rand.csv")
args = args.parse_args()

model = comet_multi_cand.load_from_checkpoint(args.model)
data = list(csv.DictReader(open(args.data)))
scores_pred = model.predict(data, batch_size=64).scores
# assume the output is always a list the size of number of additional_score_out+1
scores_pred = [x[0] for x in scores_pred]

res_pearson, res_kendall, res_errmean = utils.eval_da_per_lang(scores_pred, data)
print(f"ρ={res_pearson:.3f} τ={res_kendall:.3f} e={res_errmean:.1f}")