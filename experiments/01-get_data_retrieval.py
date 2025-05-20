import numpy as np
import sentence_transformers.util
import tqdm
import utils
import csv
import random
import os
import sentence_transformers
import argparse

args = argparse.ArgumentParser()
args.add_argument("--embd-key", default="mt", choices=["mt", "src"])
args = args.parse_args()

model = sentence_transformers.SentenceTransformer("all-MiniLM-L12-v2")

def process_data(data, data_retrieval):
    print("Computing cosine similarity")
    sims_all = sentence_transformers.util.cos_sim(
        np.array([line["embd"] for line in data]),
        np.array([line["embd"] for line in data_retrieval])
    )
    
    print("Finding nearest neighbors")
    for line, line_sim in tqdm.tqdm(zip(data, sims_all)):
        # find the most similar line that is not close to 1 (likely same text)
        line_sim[line_sim >= 0.9999] = -999
        retrieval_idx = line_sim.argsort(descending=True)[:5]
        tgts = [
            (data_retrieval[idx]["src"], data_retrieval[idx]["mt"], data_retrieval[idx]["score"])
            for idx in retrieval_idx
        ][:5]
        line["src2"], line["mt2"], line["score2"] = tgts[0]
        line["src3"], line["mt3"], line["score3"] = tgts[1]
        line["src4"], line["mt4"], line["score4"] = tgts[2]
        line["src5"], line["mt5"], line["score5"] = tgts[3]
        line["src6"], line["mt6"], line["score6"] = tgts[4]
    
    return data

def add_embd(data):
    print("Computing embeddings")
    txt_all = list({line[args.embd_key] for line in data})
    txt_to_embd = {tgt: tgt_e for tgt, tgt_e in zip(txt_all, model.encode(txt_all, show_progress_bar=True, batch_size=256))}

    for line in data:
        line["embd"] = txt_to_embd[line[args.embd_key]]

def data_flatten(data):
    # just flatten
    data_new = []
    for line in data:
        for sys in line["scores"].keys():
            data_new.append({
                "langs": line["langs"],
                "src": line["src"],
                "mt": line["tgt"][sys],
                "ref": line["ref"],
                "score": line["scores"][sys]["human"],
            })
    
    return data_new


data_train, data_test = utils.get_data()

data_train = data_flatten(data_train)
data_test = data_flatten(data_test)

add_embd(data_train)
add_embd(data_test)

data_train = process_data(data_train, data_train)
data_test = process_data(data_test, data_train)

# use 1k samples for dev
data_dev_i = random.Random(0).sample(list(range(len(data_train))), k=1_000)
data_dev = [data_train[i] for i in data_dev_i]
data_train = [data_train[i] for i in range(len(data_train)) if i not in data_dev_i]

if __name__ == "__main__":
    os.makedirs("data/csv", exist_ok=True)
    def write_data(data, split):
        print("Writing", split, "of size", str(len(data)//1000)+"k")
        with open(f"data/csv/{split}_retrieval_{args.embd_key}.csv", "w") as f:
            writer = csv.DictWriter(
                f, fieldnames=[
                    "langs",
                    "src", "ref",
                    "mt", "score",
                    "src2", "mt2", "score2",
                    "src3", "mt3", "score3",
                    "src4", "mt4", "score4",
                    "src5", "mt5", "score5",
                    "src6", "mt6", "score6",
                ])
            writer.writeheader()

            # remove extra key
            for line in data:
                line.pop("embd")
            
            writer.writerows(data)

    write_data(data_train, "train")
    write_data(data_test, "test")
    write_data(data_dev, "dev")

"""
sbatch_gpu_big_short "get_data_retrieval_src" "python3 experiments/01-get_data_retrieval.py --embd-key src"
sbatch_gpu_big_short "get_data_retrieval_mt" "python3 experiments/01-get_data_retrieval.py --embd-key mt"
"""
