import csv
import collections

import numpy as np
import utils

data = list(csv.DictReader(open("data/csv/test_same_rand.csv")))

stats = collections.Counter()
for line in data:
    num_empty = [line[f"mt{i}"] == "" for i in [2, 3, 4, 5, 6]].count(True)
    stats[5-num_empty] += 1

print(stats)

# %%

_data_train, data_test = utils.get_data()
num_additional = []
for line in data_test:
    num_additional.append(len(set(line["tgt"].values())))

print(np.average(num_additional))
print(np.median(num_additional))