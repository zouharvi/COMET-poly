import os
import glob
import re
import json
import collections
import random

os.makedirs("data/jsonl", exist_ok=True)

RANDOM_DEV = random.Random(0)
RANDOM_SHUFFLE = random.Random(0)

data = []

for dir in glob.glob("data/mt-metrics-eval-v2/*"):
    if os.path.isdir(dir):
        year = 2000+int(re.search(r"^wmt(\d{2})", dir.split("/")[-1]).group(1))
        print(dir)

        sources = {
            # drop last line which is empty
            f.split("/")[-1].split(".")[0]: open(f, "r").read().split("\n")[:-1]
            for f in glob.glob(f"{dir}/sources/*.txt")
        }
        documents = {
            # drop last line which is empty
            # take only the domain
            f.split("/")[-1].split(".")[0]: [line.split("\t") for line in open(f, "r").read().split("\n")[:-1]]
            for f in glob.glob(f"{dir}/documents/*.docs")
        }
        dir_short = dir.split("/")[-1]
        OVERRIDE_DOMAIN = (
            "flores" if "flores" in dir_short else
            "news" if "news" in dir_short else
            "tedtalks" if "tedtalks" in dir_short else
            None
        )
        def fix_domain(domain):
            if OVERRIDE_DOMAIN is not None:
                return OVERRIDE_DOMAIN
            if "wiki" in domain[1]:
                return "wiki"
            return (
                domain[0]
                .replace("conversation", "voice")
                .replace("speech", "voice")
                .replace("newstest2019", "news")
                .replace("newstest2020", "news")
                .replace("no-mqm", "old")
                .replace("ecommerce", "ec")
                .replace("user_review", "ec")
                .replace("ad", "ec")
                .replace("ec", "ecommerce")
                .replace("domain", "unknown")
                .replace("qa", "unknown")
            )
        documents = {
            langs: [fix_domain(domain) for domain in domains]
            for langs, domains in documents.items()
        }
        references = {
            # drop last line which is empty
            f.split("/")[-1].split(".")[0]: open(f, "r").read().split("\n")[:-1]
            for f in glob.glob(f"{dir}/references/*.txt")
        }
        targets = {
            # drop last line which is empty
            (langs, f.split("/")[-1].removesuffix(".txt")): open(f, "r").read().split("\n")[:-1]
            for langs in sources.keys()
            for f in glob.glob(f"{dir}/system-outputs/{langs}/*.txt")
        }

        for f in glob.glob(f"{dir}/human-scores/*.seg.score"):
            # skip MQM
            if ".mqm" in f:
                continue
            if "-z." in f:
                continue
            if ".psqm" in f:
                continue
            
            langs = f.split("/")[-1].split(".")[0]
            for line_i, line in enumerate(open(f, "r").read().split("\n")[:-1]):
                system, score = line.split()
                if score == "None":
                    continue
                try:
                    score = float(score)
                except ValueError:
                    continue

                # risky but works..
                line_i = line_i % len(sources[langs])

                data.append({
                    "src": sources[langs][line_i],
                    "ref": references[langs][line_i],
                    "tgt": targets[(langs, system)][line_i],
                    "score": score,
                    "year": year,
                    "langs": langs,
                    "system": system,
                    "domain": documents[langs][line_i],
                })

with open("data/jsonl/all.jsonl", "w") as f:
    f.writelines([json.dumps(line, ensure_ascii=False) + "\n" for line in data])

# we're removing only about 10 examples
data_train = [x for x in data if x["year"] <= 2021 if len(x["src"]+x["tgt"]+x["ref"]) < 2500]
data_test = [x for x in data if x["year"] == 2023]

data_train_by_langs = collections.defaultdict(list)
for x in data_train:
    data_train_by_langs[x["langs"]].append(x)
data_dev = []

for lang in set(x["langs"] for x in data_test):
    if lang not in data_train_by_langs:
        continue
    for _ in range(1000):
        data_dev.append(data_train_by_langs[lang].pop(RANDOM_DEV.randint(0, len(data_train_by_langs[lang])-1)))

# flatten and shuffle
data_train = [
    x
    for v in data_train_by_langs.values()
    for x in v
]
RANDOM_SHUFFLE.shuffle(data_train)

print("TRAIN:", len(data_train))
print("TEST: ", len(data_test))
print("DEV:  ", len(data_dev))

open("data/jsonl/train.jsonl", "w").writelines(json.dumps(line, ensure_ascii=False) + "\n" for line in data_train)
open("data/jsonl/test.jsonl", "w").writelines(json.dumps(line, ensure_ascii=False) + "\n" for line in data_test)
open("data/jsonl/dev.jsonl", "w").writelines(json.dumps(line, ensure_ascii=False) + "\n" for line in data_dev)
