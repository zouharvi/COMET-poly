import itertools
import numpy as np
import comet
import argparse
import json
import collections
import random

r_anchor = random.Random(0)

args = argparse.ArgumentParser()
args.add_argument("model")
args = args.parse_args()

model = comet.load_from_checkpoint(args.model)

# load data
data = [json.loads(x) for x in open("data/jsonl/test.jsonl")]
src_to_tgts = collections.defaultdict(list)
for x in data:
    src_to_tgts[(x["src"], x["langs"])].append((x["tgt"], x["score"]))
src_to_tgts = {
    src: sorted(tgts, key=lambda x: x[1], reverse=True)
    for src, tgts in src_to_tgts.items()
}
data1 = []
data3 = []
for (src, _langs), tgts in src_to_tgts.items():
    r_anchor.shuffle(tgts)
    # make sure the translations are unique
    tgts = [x for i, x in enumerate(tgts) if x[0] not in {x[0] for x in tgts[:i]}]
    
    # get all triplets
    for l1, l2, l3 in itertools.combinations(tgts, 3):
        mt1, score1 = l1
        mt3, score3 = l3
        # anchor
        mt2, score2 = l2

        # score1 is always higher than score3
        # the difference is at least 10
        if not (score1 > score3 + 10):
            continue

        data1.append({
            "src": src,
            "mt": mt1,
            "score": score1,
            "mt2": mt2,
            "score2": score2,
        })
        data3.append({
            "src": src,
            "mt": mt3,
            "score": score3,
            "mt2": mt2,
            "score2": score2,
        })
print(f"Data len: {len(data1)}")

# TODO: use all data
scores1_pred = model.predict(random.Random(0).sample(data1, k=10_000), batch_size=32).scores
scores3_pred = model.predict(random.Random(0).sample(data3, k=10_000), batch_size=32).scores

accuracy_pw = np.average([s1 > s3 for s1, s3 in zip(scores1_pred, scores3_pred)])

print(f"Pairwise accuracy:  {accuracy_pw:.2%}")

"""
sbatch_gpu_short "10pw_eval_anchor-baseline"     "python3 experiments/10pw-eval_anchor.py /cluster/work/sachan/vilem/COMET-ranking/lightning_logs/version_2/checkpoints/epoch=4-step=26540-val_kendall=0.297.ckpt"
sbatch_gpu_short "10pw_eval_anchor-metric"       "python3 experiments/10pw-eval_anchor.py /cluster/work/sachan/vilem/COMET-ranking/lightning_logs/version_0/checkpoints/epoch=4-step=26540-val_kendall=0.306.ckpt"
sbatch_gpu_short "10pw_eval_anchor-score_metric" "python3 experiments/10pw-eval_anchor.py /cluster/work/sachan/vilem/COMET-ranking/lightning_logs/version_1/checkpoints/epoch=4-step=26540-val_kendall=0.356.ckpt"
"""