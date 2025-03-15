import utils
import csv
import random
import os

def process_data(data):
    data_new = []
    r = random.Random(0)

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
            # fill up to 5 with empty string and 0 as score
            tgts = tgts[:5]
            tgts += [("", 0)] * (5 - len(tgts))

            data_new.append({
                "src": line["src"],
                "mt": line["tgt"][sys],
                "score": line["scores"][sys]["human"],
                **{
                    f"mt{i+2}": mt for i, (mt, _) in enumerate(tgts)
                },
                **{
                    f"score{i+2}": score for i, (_, score) in enumerate(tgts)
                }
            })
    return data_new
    

data_train, data_test = utils.get_data()
data_train = process_data(data_train)
data_test = process_data(data_test)

# use 1k samples for dev
data_dev_i = random.Random(0).sample(list(range(len(data_train))), k=1_000)
data_dev = [data_train[i] for i in data_dev_i]
data_train = [data_train[i] for i in range(len(data_train)) if i not in data_dev_i]


if __name__ == "__main__":
    os.makedirs("data/csv", exist_ok=True)
    def write_data(data, split):
        print("Writing", split, "of size", str(len(data)//1000)+"k")
        with open(f"data/csv/{split}_multi.csv", "w") as f:
            writer = csv.DictWriter(
                f, fieldnames=[
                    "src", "mt", "score",
                    "mt2", "score2",
                    "mt3", "score3",
                    "mt4", "score4",
                    "mt5", "score5",
                    "mt6", "score6",
                ])
            writer.writeheader()
            writer.writerows(data)

    write_data(data_train, "train")
    write_data(data_dev, "dev")
    write_data(data_test, "test")