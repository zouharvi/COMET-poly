# %%

import csv
import numpy as np
import utils

data_multi_sim = list(csv.DictReader(open("data/csv/test_same_sim.csv")))
data_multi_ran = list(csv.DictReader(open("data/csv/test_same_rand.csv")))

score_avg = np.average([float(x["score"]) for x in data_multi_sim])

# closest
def pred_score_v1(line):
    return float(line["score2"])

# closest (avg)
def pred_score_v2(line):
    scores = [line["score2"], line["score3"], line["score4"], line["score5"], line["score6"]]
    scores = [float(x) for x in scores if x != "0"]
    if not scores:
        return score_avg
    return np.average(scores)

def eval_baseline(name, fn, data):
    scores_pred = [fn(x) for x in data]
    res_pearson, res_kendall, res_errmean = utils.eval_da_per_lang(scores_pred, data)
    print(f"{name:>20} ρ={res_pearson:.3f} τ={res_kendall:.3f} e={res_errmean:.1f}")

eval_baseline("closest", pred_score_v1, data_multi_sim)
eval_baseline("closest-avg", pred_score_v2, data_multi_sim)
eval_baseline("random", pred_score_v1, data_multi_ran)
eval_baseline("random-avg", pred_score_v2, data_multi_ran)