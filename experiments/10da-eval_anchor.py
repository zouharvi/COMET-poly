import comet
import argparse
import csv
import scipy.stats

args = argparse.ArgumentParser()
args.add_argument("model")
args = args.parse_args()

model = comet.load_from_checkpoint(args.model)

# load data
data = list(csv.DictReader(open("data/csv/test_anchor.csv")))
scores_pred = model.predict(data, batch_size=32).scores
scores_true = [float(x["score"]) for x in data]

print(f"Pearson:  {scipy.stats.pearsonr(scores_true, scores_pred)[0]:.3f}")
print(f"Kendal-b: {scipy.stats.kendalltau(scores_true, scores_pred, variant='b')[0]:.3f}")

"""
sbatch_gpu_short "10da_eval_anchor-baseline"     "python3 experiments/10da-eval_anchor.py /cluster/work/sachan/vilem/COMET-ranking/lightning_logs/version_2/checkpoints/epoch=4-step=26540-val_kendall=0.297.ckpt"
sbatch_gpu_short "10da_eval_anchor-metric"       "python3 experiments/10da-eval_anchor.py /cluster/work/sachan/vilem/COMET-ranking/lightning_logs/version_0/checkpoints/epoch=4-step=26540-val_kendall=0.306.ckpt"
sbatch_gpu_short "10da_eval_anchor-score_metric" "python3 experiments/10da-eval_anchor.py /cluster/work/sachan/vilem/COMET-ranking/lightning_logs/version_1/checkpoints/epoch=4-step=26540-val_kendall=0.356.ckpt"
"""