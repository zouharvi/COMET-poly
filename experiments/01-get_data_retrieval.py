import numpy as np
import sentence_transformers.util
import torch
import tqdm
import utils
import csv
import random
import os
import sentence_transformers
import argparse
from annoy import AnnoyIndex

args = argparse.ArgumentParser()
args.add_argument("--embd-key", default="mt", choices=["mt", "src", "srcAmt", "srcCmt"])
args = args.parse_args()

model = sentence_transformers.SentenceTransformer("all-MiniLM-L12-v2")

def process_data(data, data_retrieval, data_retrieval_index: AnnoyIndex):
    print("Finding nearest neighbors")
    for line in tqdm.tqdm(data):
        # we must select much more (n) because we are filtering it later on
        idx, dists = data_retrieval_index.get_nns_by_vector(line["embd"], n=500, include_distances=True)

        tgts = [
            (data_retrieval[i]["src"], data_retrieval[i]["mt"], data_retrieval[i]["score"])
            for i in idx
        ]
        # filter by hard match instead of inner product
        # idx = [i for i, d in zip(idx, dists) if d < 0.999999][:5]
        tgts = [
            x for x in tgts
            if (x[0], x[1]) != (line["src"], line["mt"])
        ][:5]
        line["src2"], line["mt2"], line["score2"] = tgts[0]
        line["src3"], line["mt3"], line["score3"] = tgts[1]
        line["src4"], line["mt4"], line["score4"] = tgts[2]
        line["src5"], line["mt5"], line["score5"] = tgts[3]
        line["src6"], line["mt6"], line["score6"] = tgts[4]
    
    return data

def add_embd(data):
    print("Computing embeddings")

    if args.embd_key == "mt":
        tgt_all = list({line["mt"] for line in data})
        tgt_to_embd = {
            tgt: tgt_e
            for tgt, tgt_e in zip(tgt_all, model.encode(tgt_all, show_progress_bar=True, batch_size=256, normalize_embeddings=True))
        }
        for line in data:
            line["embd"] = tgt_to_embd[line["mt"]]
    elif args.embd_key == "src":
        src_all = list({line["src"] for line in data})
        src_to_embd = {
            tgt: tgt_e
            for tgt, tgt_e in zip(src_all, model.encode(src_all, show_progress_bar=True, batch_size=256, normalize_embeddings=True))
        }
        for line in data:
            line["embd"] = src_to_embd[line["src"]]
    elif args.embd_key == "srcAmt":
        src_all = list({line["src"] for line in data})
        tgt_all = list({line["mt"] for line in data})
        tgt_to_embd = {
            tgt: tgt_e
            for tgt, tgt_e in zip(tgt_all, model.encode(tgt_all, show_progress_bar=True, batch_size=256))
        }
        src_to_embd = {
            tgt: tgt_e
            for tgt, tgt_e in zip(src_all, model.encode(src_all, show_progress_bar=True, batch_size=256))
        }
        for line in data:
            line["embd"] = src_to_embd[line["src"]] + tgt_to_embd[line["mt"]]
    elif args.embd_key == "srcCmt":
        src_all = list({line["src"] for line in data})
        tgt_all = list({line["mt"] for line in data})
        tgt_to_embd = {
            tgt: tgt_e
            for tgt, tgt_e in zip(tgt_all, model.encode(tgt_all, show_progress_bar=True, batch_size=256))
        }
        src_to_embd = {
            tgt: tgt_e
            for tgt, tgt_e in zip(src_all, model.encode(src_all, show_progress_bar=True, batch_size=256))
        }
        for line in data:
            line["embd"] = np.concatenate((src_to_embd[line["src"]], tgt_to_embd[line["mt"]]))
    else:
        raise ValueError("Unknown embedding key")
    
    # normalize embeddings
    for line in data:
        line["embd"] = line["embd"] / np.linalg.norm(line["embd"])

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

print("Populating Annoy")
data_train_index = AnnoyIndex(len(data_train[0]["embd"]), 'dot')
for i, line in enumerate(data_train):
    data_train_index.add_item(i, line["embd"])
print("Buliding Annoy")
data_train_index.build(20)

data_train = process_data(data_train, data_train, data_train_index)
data_test = process_data(data_test, data_train, data_train_index)

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
function sbatch_gpu_short() {
    JOB_NAME=$1;
    JOB_WRAP=$2;
    mkdir -p logs

    sbatch \
        -J $JOB_NAME --output=logs/%x.out --error=logs/%x.err \
        --gpus=1 --gres=gpumem:22g \
        --mail-type END \
        --mail-user vilem.zouhar@gmail.com \
        --ntasks-per-node=1 \
        --cpus-per-task=12 \
        --mem-per-cpu=14G --time=0-4 \
        --wrap="$JOB_WRAP";
}


sbatch_gpu_short "get_data_retrieval_src" "python3 experiments/01-get_data_retrieval.py --embd-key src"
sbatch_gpu_short "get_data_retrieval_srcAmt" "python3 experiments/01-get_data_retrieval.py --embd-key srcAmt"
sbatch_gpu_short "get_data_retrieval_srcCmt" "python3 experiments/01-get_data_retrieval.py --embd-key srcCmt"
sbatch_gpu_short "get_data_retrieval_mt" "python3 experiments/01-get_data_retrieval.py --embd-key mt"
"""
