import comet_multi_cand
import argparse
import csv
import utils
import json

args = argparse.ArgumentParser()
args.add_argument("model")
args.add_argument("t", type=int)
args = args.parse_args()

model = comet_multi_cand.load_from_checkpoint(args.model)
data = list(csv.DictReader(open("data/csv/test_same_rand.csv")))
for line in data:
    line["mt2"], line["score2"] = line[f"mt{args.t}"], line[f"score{args.t}"]

scores_pred = model.predict(data, batch_size=64).scores
# assume the output is always a list the size of number of additional_score_out+1
scores_pred = [x[0] for x in scores_pred]

res_pearson, res_kendall, res_errmean = utils.eval_da_per_lang(scores_pred, data)
print(json.dumps({"pearson": res_pearson, "kendall": res_kendall, "meanerr": res_errmean, "t": args.t, "model": args.model}))

"""
sbatch_gpu_short "mazurek_0t00s_t2" "python3 experiments/21-compute_mazurek.py lightning_logs/multicand_0t00s/checkpoints/epoch\=4* 2"
for t in 2 3 4 5 6; do
    sbatch_gpu_short "mazurek_1t00s_t${t}" "python3 experiments/21-compute_mazurek.py lightning_logs/multicand_1t00s/checkpoints/epoch\=4* ${t}"
    sbatch_gpu_short "mazurek_1t10s_t${t}" "python3 experiments/21-compute_mazurek.py lightning_logs/multicand_1t10s/checkpoints/epoch\=4* ${t}"
    sbatch_gpu_short "mazurek_1t01s_t${t}" "python3 experiments/21-compute_mazurek.py lightning_logs/multicand_1t01s/checkpoints/epoch\=4* ${t}"
done;

# previously this experiment was called "nebel"
cat logs/nebel_*.out > logs/mazurek.out
scp euler:/cluster/work/sachan/vilem/COMET-multi-cand/logs/mazurek.out computed/mazurek.out
"""