# %%
import utils_fig
import matplotlib.pyplot as plt
import json
import collections
import re

data_raw = [json.loads(x) for x in open("../computed/nebel.out", "r")]
data_raw.sort(key=lambda x: x["t"])

data = collections.defaultdict(list)
for line in data_raw:
    data[line["model"]].append(line)

def get_model_name(model):
    if "0t00" in model:
        return "Standard $f(s, t) \\rightarrow \\hat{y}$"
    elif "1t00" in model:
        return "Additional translation $f(s, t, t_2) \\rightarrow \\hat{y}$"
    elif "1t01" in model:
        return "Additional translation & output $f(s, t, t_2) \\rightarrow \\hat{y}, \\hat{y_{t_2}}$"
    elif "1t10" in model:
        return "Additional translation & its score $f(s, t, t_2, y_{t_2}) \\rightarrow \\hat{y}$"

def get_model_marker(model):
    if "1t00" in model:
        return "s"
    elif "1t01" in model:
        return "^"
    elif "1t10" in model:
        return "o"

_, axs = plt.subplots(1, 3, figsize=(9, 2))
for ax, key in zip(axs, ["pearson", "kendall", "meanerr"]):
    for model, model_v in data.items():
        if "0t00" in model:
            ax.hlines(
                y=[x[key] for x in model_v],
                xmin=2,
                xmax=6,
                linewidth=2,
                label=get_model_name(model),
                linestyle="--",
                color="black",
            )
        else:
            
            ax.plot(
                [x["t"] for x in model_v],
                [x[key] for x in model_v],
                label=get_model_name(model),
                marker=get_model_marker(model),
                linewidth=0,
                markersize=10,
            )
    # ax.legend()
    ax.set_title({
        "pearson": "Pearson $\\rho \\uparrow$",
        "kendall": "Kendall $\\tau \\uparrow$",
        "meanerr": "MAE $\\downarrow$",
    }[key], pad=-10)
    ax.set_xlabel("Additional translation is {x} closest")

    ax.set_xticks([2, 3, 4, 5, 6])
    ax.set_xticklabels(["1st", "2nd", "3rd", "4th", "5th"])

axs[0].set_ylim(0.05, 0.25)
axs[1].set_ylim(0.05, 0.25)
axs[2].set_ylim(22, 31)

utils_fig.turn_off_spines(ax=axs[0])
utils_fig.turn_off_spines(ax=axs[1])
utils_fig.turn_off_spines(ax=axs[2])

plt.tight_layout(pad=0)
plt.subplots_adjust(wspace=0.3)
plt.savefig("../figures/22-nebel.pdf")
plt.show()

# %%

# replicate legend based on ax 
plt.figure(figsize=(8, 0.7))
handles, labels = ax.get_legend_handles_labels()
plt.legend(
    handles,
    labels,
    loc="center",
    ncol=2,
    frameon=False,
    fontsize=9,
    handletextpad=0.1,
)

plt.axis("off")

plt.tight_layout(pad=0)
plt.savefig("../figures/22-nebel_legend.pdf")
plt.show()