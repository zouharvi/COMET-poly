import comet_multi_cand
import argparse
import csv

args = argparse.ArgumentParser()
args.add_argument("--sim", action="store_true")
args.add_argument("model")
args = args.parse_args()

model = comet_multi_cand.load_from_checkpoint(args.model)
if args.sim:
    data = list(csv.DictReader(open("data/csv/test_same_sim.csv")))
else:
    data = list(csv.DictReader(open("data/csv/test_same_rand.csv")))
scores_pred = model.predict(data, batch_size=64).scores
# assume the output is always a list the size of number of additional_score_out+1
scores_pred = [x[0] for x in scores_pred]

print("score_model")
for score in scores_pred:
    print(score)

"""
sbatch_gpu_short "output_simple_0t00s" "python3 experiments/02-output_simple.py lightning_logs/multicand_0t00s/checkpoints/epoch\=4*"
"""