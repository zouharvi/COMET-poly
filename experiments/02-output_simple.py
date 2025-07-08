import comet_poly
import argparse
import csv

args = argparse.ArgumentParser()
args.add_argument("--sim", action="store_true")
args.add_argument("model")
args = args.parse_args()

model = comet_poly.load_from_checkpoint(args.model)
if args.sim:
    data = list(csv.DictReader(open("data/csv/test_same_sim.csv")))
else:
    data = list(csv.DictReader(open("data/csv/test_same_rand.csv")))
scores_pred = model.predict(data, batch_size=64).scores

# now reduce scores_pred for eval
if type(scores_pred[0]) is list:
    scores_pred = [x[0] for x in scores_pred]

print("score_model")
for score in scores_pred:
    print(score)

"""
sbatch_gpu_short "output_simple_0t00s" "python3 experiments/02-output_simple.py lightning_logs/polycand_0t00s/checkpoints/epoch\=4*"
"""