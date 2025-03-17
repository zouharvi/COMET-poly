# %%
import utils_fig
import matplotlib.pyplot as plt
import json
import collections

data_raw = [json.loads(x) for x in open("../computed/nebel.out", "r")]
data_raw.sort(key=lambda x: x["t"])

data = collections.defaultdict(list)
for line in data_raw:
    data[line["model"]].append(line)

_, axs = plt.subplots(1, 3, figsize=(9, 2))
for ax, key in zip(axs, ["pearson", "kendall", "meanerr"]):
    for model, model_v in data.items():
        ax.plot(
            [x["t"] for x in model_v],
            [x[key] for x in model_v],
            label=model
        )
    # ax.legend()
    ax.set_ylabel(key)
    ax.set_xlabel("t")

plt.tight_layout(pad=0)
plt.savefig("../figures/nebel.pdf")
plt.show()