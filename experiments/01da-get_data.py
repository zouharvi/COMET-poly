import utils
import csv
import random

def process_data(data):
    data_new = []

    # just flatten
    data_new = []
    for line in data:
        for sys in line["scores"].keys():
            data_new.append({
                "src": line["src"],
                "mt": line["tgt"][sys],
                "score": line["scores"][sys]["human"],
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
    def write_data(data, split):
        print("Writing", split, "of size", str(len(data)//1000)+"k")
        with open(f"data/csv/{split}_da.csv", "w") as f:
            writer = csv.DictWriter(f, fieldnames=["src", "mt", "score"])
            writer.writeheader()
            writer.writerows(data)

    write_data(data_train, "train")
    write_data(data_dev, "dev")
    write_data(data_test, "test")