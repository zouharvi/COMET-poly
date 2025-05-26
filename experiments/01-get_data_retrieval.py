from typing import List, Tuple
import numpy as np
import tqdm
import utils
import csv
import random
import os
import argparse
from annoy import AnnoyIndex

args = argparse.ArgumentParser()
args.add_argument("--embd-key", default="mt", choices=["mt", "src", "srcAmt", "srcCmt"])
args.add_argument("--embd-model", default="minilm", choices=["minilm", "xlmr", "comet"])
args.add_argument("--max-sim", choices=["11", "07", "05"], default="11")
args = args.parse_args()

max_sim = {
    # no similarity filtering because max is 1.0
    "11": 1.1,
    "07": 0.7,
    "05": 0.5,
}[args.max_sim]

R_FALLBACK = random.Random(0)

def process_data(data, data_retrieval, data_retrieval_index: AnnoyIndex, prevent_hardmatch=False) -> Tuple[List[dict], List[List[float]]]:
    print("Finding nearest neighbors")
    data_sim = []
    for line in tqdm.tqdm(data):
        # select increasingly large neighbours because we filter later on
        for n in [50, 100, 500, 1000, 5000]:
            idx, dists = data_retrieval_index.get_nns_by_vector(line["embd"], n=n, include_distances=True)
            idx = [(i, d) for i, d in zip(idx, dists) if d <= max_sim or not prevent_hardmatch]
            tgts = [
                (data_retrieval[i]["src"], data_retrieval[i]["mt"], data_retrieval[i]["score"], d)
                for i, d in idx
            ]
            if prevent_hardmatch:
                tgts = [
                    x for x in tgts
                    if x[0] != line["src"] and x[1] != line["mt"]
                ]
            tgts = tgts[:5]
            if len(tgts) == 5:
                break
        else:
            # just take random examples at this point
            tgts += [
                (line["src"], line["mt"], line["score"], None)
                for line in R_FALLBACK.sample(data_retrieval, 5-len(tgts))
            ]

        line["src2"], line["mt2"], line["score2"], sim2 = tgts[0]
        line["src3"], line["mt3"], line["score3"], sim3 = tgts[1]
        line["src4"], line["mt4"], line["score4"], sim4 = tgts[2]
        line["src5"], line["mt5"], line["score5"], sim5 = tgts[3]
        line["src6"], line["mt6"], line["score6"], sim6 = tgts[4]

        data_sim.append([sim2, sim3, sim4, sim5, sim6])
    
    return data, data_sim

def encode_txts(txts):

    if args.embd_model in {"xlmr", "minilm"}:
        import sentence_transformers
        if args.embd_model == "xlmr":
            model = sentence_transformers.SentenceTransformer("stsb-xlm-r-multilingual")
        elif args.embd_model == "minilm":
            model = sentence_transformers.SentenceTransformer("all-MiniLM-L12-v2")
        
        return {
            txt: txt_e
            for txt, txt_e in zip(txts, model.encode(txts, show_progress_bar=True, batch_size=256))
        }
    elif args.embd_model == "comet":
        import comet
        model = comet.load_from_checkpoint(comet.download_model("Unbabel/wmt22-comet-da")).to("cuda:0")
        samples = {}
        for txts_batch in tqdm.tqdm([txts[i:i+128] for i in range(0, len(txts), 128)]):
            ids = model.encoder.prepare_sample(txts_batch).to("cuda:0")
            ids = {
                k: v.to("cuda:0")
                for k, v in ids.items()
            }
            samples |= {
                txt: txt_e
                for txt, txt_e in zip(txts_batch, model.encoder(**ids)["sentemb"].cpu().numpy())
            }
        return samples


def add_embd(data):
    print("Computing embeddings")

    if args.embd_key == "mt":
        tgt_all = list({line["mt"] for line in data})
        tgt_to_embd = encode_txts(tgt_all)
        for line in data:
            line["embd"] = tgt_to_embd[line["mt"]]
    elif args.embd_key == "src":
        src_all = list({line["src"] for line in data})
        src_to_embd = encode_txts(src_all)
        for line in data:
            line["embd"] = src_to_embd[line["src"]]
    elif args.embd_key == "srcAmt":
        src_all = list({line["src"] for line in data})
        tgt_all = list({line["mt"] for line in data})
        tgt_to_embd = encode_txts(tgt_all)
        src_to_embd = encode_txts(src_all)
        for line in data:
            line["embd"] = src_to_embd[line["src"]] + tgt_to_embd[line["mt"]]
    elif args.embd_key == "srcCmt":
        src_all = list({line["src"] for line in data})
        tgt_all = list({line["mt"] for line in data})
        tgt_to_embd = encode_txts(tgt_all)
        src_to_embd = encode_txts(src_all)
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
data_train_index.build(50)

data_train, sim_train = process_data(data_train, data_train, data_train_index, prevent_hardmatch=True)
# for test, we don't mind at all if we retrieve the same translation
data_test, sim_test = process_data(data_test, data_train, data_train_index, prevent_hardmatch=False)

# use 1k samples for dev
data_dev_i = random.Random(0).sample(list(range(len(data_train))), k=1_000)
data_dev = [data_train[i] for i in data_dev_i]
sim_dev = [sim_train[i] for i in data_dev_i]
data_train = [data_train[i] for i in range(len(data_train)) if i not in data_dev_i]
sim_train = [sim_train[i] for i in range(len(sim_train)) if i not in data_dev_i]

os.makedirs("data/csv", exist_ok=True)
def write_data(data, split):
    print("Writing", split, "of size", str(len(data)//1000)+"k")
    with open(f"data/csv/{split}_retrieval_{args.embd_model}_{args.max_sim}_{args.embd_key}.csv", "w") as f:
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


def write_sim(data, split):
    with open(f"computed/sim_{split}_retrieval_{args.embd_model}_{args.max_sim}_{args.embd_key}.npy", "wb") as f:
        np.save(f, np.array(data))

write_sim(sim_test, "test")
write_sim(sim_dev, "dev")

"""
mkdir -p computed

# for testing
python3 experiments/01-get_data_retrieval.py --embd-key src    --embd-model minilm --max-sim 11
python3 experiments/01-get_data_retrieval.py --embd-key srcAmt
python3 experiments/01-get_data_retrieval.py --embd-key srcCmt
python3 experiments/01-get_data_retrieval.py --embd-key mt
python3 experiments/01-get_data_retrieval.py --embd-key srcCmt --embd-model comet --max-sim 07
python3 experiments/01-get_data_retrieval.py --embd-model xlmr


function sbatch_gpu() {
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
        --mem-per-cpu=10G --time=1-0 \
        --wrap="$JOB_WRAP";
}

for MAXSIM in "11" "07" "05"; do
for EMBDKEY in "src" "srcAmt" "srcCmt" "mt"; do
for EMBDMODEL in "minilm" "xlmr" "comet"; do
    sbatch_gpu "get_data_retrieval_${EMBDMODEL}_${MAXSIM}_${EMBDKEY}" "python3 experiments/01-get_data_retrieval.py --embd-key $EMBDKEY --embd-model $EMBDMODEL --max-sim $MAXSIM"
done;
done;
done;
"""
