import sentence_transformers.util
import tqdm
import utils
import csv
import random
import os
import sentence_transformers
import argparse

args = argparse.ArgumentParser()
args.add_argument("--data-name", default="wmt", choices=["wmt", "bio"]) 
args.add_argument("--sort_by_sim", action="store_true")
args = args.parse_args()

model = sentence_transformers.SentenceTransformer("all-MiniLM-L12-v2")

def process_data(data):
    data_new = []
    r = random.Random(0)

    tgt_all = set()

    # just flatten
    data_new = []
    for line in data:
        for sys in line["scores"].keys():
            tgts = [
                (line["tgt"][sys2], line["scores"][sys2]["human"])
                for sys2 in line["scores"].keys()
                if line["tgt"][sys] != line["tgt"][sys2]
            ]
            r.shuffle(tgts)
            # will be sorted by embedding similarity later if --sort-by-sim

            # fill up to 5 with empty string and 0 as score
            tgts += [("", 0)] * (5 - len(tgts))

            tgt_all.update([tgt for tgt, _ in tgts])
            tgt_all.add(line["tgt"][sys])

            data_new.append({
                "langs": line["langs"],
                "src": line["src"],
                "mt": line["tgt"][sys],
                "ref": line["ref"],
                "score": line["scores"][sys]["human"],
                "tgts": tgts,
            })

    print("Computing embeddings")
    tgt_all = list(tgt_all)
    tgt_to_embd = {tgt: tgt_e for tgt, tgt_e in zip(tgt_all, model.encode(tgt_all, show_progress_bar=True, batch_size=256))}

    for line in tqdm.tqdm(data_new):
        tgts = line.pop("tgts")
        tgts = [
            (tgt, score, sentence_transformers.util.cos_sim(tgt_to_embd[tgt], tgt_to_embd[line["mt"]]).item())
            for tgt, score in tgts
        ]
        if args.sort_by_sim:
            tgts.sort(key=lambda x: x[2], reverse=True)
        tgts = tgts[:5]
        line["mt2"], line["score2"], _ = tgts[0]
        line["mt3"], line["score3"], _ = tgts[1]
        line["mt4"], line["score4"], _ = tgts[2]
        line["mt5"], line["score5"], _ = tgts[3]
        line["mt6"], line["score6"], _ = tgts[4]

    return data_new
    

data_train, data_test = utils.get_data(data_name=args.data_name)
data_test = process_data(data_test)
data_train = process_data(data_train)

# use 1k samples for dev
data_dev_i = random.Random(0).sample(list(range(len(data_train))), k=1_000)
data_dev = [data_train[i] for i in data_dev_i]
# only filter dev for WMT where we have a lot of data
if args.data_name == "wmt":
    data_train = [data_train[i] for i in range(len(data_train)) if i not in data_dev_i]


if __name__ == "__main__":
    os.makedirs("data/csv", exist_ok=True)
    def write_data(data, split):
        print("Writing", split, "of size", str(len(data)//1000)+"k")
        data_name_prefix = f"{args.data_name}_" if args.data_name != "wmt" else ""
        if args.sort_by_sim:
            fname = f"{split}_{data_name_prefix}same_sim.csv"
        else:
            fname = f"{split}_{data_name_prefix}same_rand.csv"
        with open(f"data/csv/{fname}", "w") as f:
            writer = csv.DictWriter(
                f, fieldnames=[
                    "langs",
                    "src", "ref",
                    "mt", "score",
                    "mt2", "score2",
                    "mt3", "score3",
                    "mt4", "score4",
                    "mt5", "score5",
                    "mt6", "score6",
                ])
            writer.writeheader()
            writer.writerows(data)

    write_data(data_train, "train")
    write_data(data_test, "test")
    write_data(data_dev, "dev")




"""
sbatch_gpu "01-get_data_same_wmt_rand" "python3 experiments/01-get_data_same.py --data-name wmt"
sbatch_gpu "01-get_data_same_wmt_sim" "python3 experiments/01-get_data_same.py --data-name wmt --sort_by_sim"
sbatch_gpu "01-get_data_same_bio_rand" "python3 experiments/01-get_data_same.py --data-name bio"
sbatch_gpu "01-get_data_same_bio_sim" "python3 experiments/01-get_data_same.py --data-name bio --sort_by_sim"
"""
