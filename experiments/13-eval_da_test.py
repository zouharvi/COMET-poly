import comet_multi_cand
import argparse
import csv
import utils
import os

args = argparse.ArgumentParser()
args.add_argument("model")
args.add_argument("--data", default="data/csv/test_same_rand.csv")
args.add_argument("--output_hyp", action="store_true")

args = args.parse_args()

model = comet_multi_cand.load_from_checkpoint(args.model)
data = list(csv.DictReader(open(args.data)))
scores_pred = model.predict(data, batch_size=64).scores

if args.output_hyp:
    # Extract model name by removing 'lightning_logs/' and everything after 'checkpoints/'
    model_name = args.model.split("/checkpoints")[0]
    data_name = args.data.split(".csv")[0].replace("/", "_")
    output_filename = f"{model_name}/hyps_{data_name}.csv"

    with open(output_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["score"])
        for x in scores_pred:
            writer.writerow([x])

    print("Hypthoeses stored in ", output_filename)

# Now reduce scores_pred for eval
scores_pred = [x[0] for x in scores_pred]

res_pearson, res_kendall, res_errmean = utils.eval_da_per_lang(scores_pred, data)
print(f"ρ={res_pearson:.3f} τ={res_kendall:.3f} e={res_errmean:.1f}")
