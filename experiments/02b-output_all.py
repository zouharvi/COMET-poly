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
scores_pred_t1 = [x[0] for x in model.predict(data, batch_size=64).scores]
scores_pred_t2 = [x[0] for x in model.predict([{"src": l["src"], "mt": l["mt2"]} for l in data], batch_size=64).scores]
scores_pred_t3 = [x[0] for x in model.predict([{"src": l["src"], "mt": l["mt3"]} for l in data], batch_size=64).scores]
scores_pred_t4 = [x[0] for x in model.predict([{"src": l["src"], "mt": l["mt4"]} for l in data], batch_size=64).scores]
scores_pred_t5 = [x[0] for x in model.predict([{"src": l["src"], "mt": l["mt5"]} for l in data], batch_size=64).scores]
scores_pred_t6 = [x[0] for x in model.predict([{"src": l["src"], "mt": l["mt6"]} for l in data], batch_size=64).scores]

print("score_model,score2_model,score3_model,score4_model,score5_model,score6_model")
for score1, score2, score3, score4, score5, score6 in zip(scores_pred_t1, scores_pred_t2, scores_pred_t3, scores_pred_t4, scores_pred_t5, scores_pred_t6):
    print(score1, score2, score3, score4, score5, score6, sep=",")

"""
sbatch_gpu_short "output_all_0t00s" "python3 experiments/02b-output_all.py lightning_logs/multicand_0t00s/checkpoints/epoch\=4*"
"""