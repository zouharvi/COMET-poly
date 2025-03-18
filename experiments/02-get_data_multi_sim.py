import sentence_transformers.util
import tqdm
import utils
import csv
import random
import os
import sentence_transformers

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
            # will be sorted by embedding similarity later

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
        tgts.sort(key=lambda x: x[2], reverse=True)
        tgts = tgts[:5]
        line["mt2"], line["score2"], line["sim2"] = tgts[0]
        line["mt3"], line["score3"], line["sim3"] = tgts[1]
        line["mt4"], line["score4"], line["sim4"] = tgts[2]
        line["mt5"], line["score5"], line["sim5"] = tgts[3]
        line["mt6"], line["score6"], line["sim6"] = tgts[4]

    return data_new
    

data_train, data_test = utils.get_data()
data_test = process_data(data_test)
data_train = process_data(data_train)

# use 1k samples for dev
data_dev_i = random.Random(0).sample(list(range(len(data_train))), k=1_000)
data_dev = [data_train[i] for i in data_dev_i]
data_train = [data_train[i] for i in range(len(data_train)) if i not in data_dev_i]


if __name__ == "__main__":
    os.makedirs("data/csv", exist_ok=True)
    def write_data(data, split):
        print("Writing", split, "of size", str(len(data)//1000)+"k")
        with open(f"data/csv/{split}_multi_sim.csv", "w") as f:
            writer = csv.DictWriter(
                f, fieldnames=[
                    "langs",
                    "src", "ref",
                    "mt", "score",
                    "mt2", "score2", "sim2",
                    "mt3", "score3", "sim3",
                    "mt4", "score4", "sim4",
                    "mt5", "score5", "sim5",
                    "mt6", "score6", "sim6",
                ])
            writer.writeheader()
            writer.writerows(data)

    write_data(data_train, "train")
    write_data(data_test, "test")
    write_data(data_dev, "dev")