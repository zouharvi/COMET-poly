# %%

"""
rsync -azP  euler:/cluster/work/sachan/vilem/COMET-poly/computed/sim_*.npy computed/
"""

import numpy as np
import matplotlib.pyplot as plt

THRESHOLD = "11"
EMBEDDING = "mt"

sim_dev = np.load(f"../computed/sim_dev_retrieval_minilm_{THRESHOLD}_{EMBEDDING}.npy", allow_pickle=True)
sim_test = np.load(f"../computed/sim_test_retrieval_minilm_{THRESHOLD}_{EMBEDDING}.npy", allow_pickle=True)


sim_dev[sim_dev == None] = 0
sim_test[sim_test == None] = 0

fig, axs = plt.subplots(5, 2, figsize=(4, 8), sharex=True, sharey=True)

for i in range(5):
    axs[i,0].hist(
        sim_dev[:,i],
        bins=np.linspace(0.0, 1.0, 100),
        density=True,
    )
    axs[i,1].hist(
        sim_test[:,i],
        bins=np.linspace(0.0, 1.0, 100),
        density=True,
    )

    axs[i,0].set_title(f"Train {i+1}th neighbor")
    axs[i,1].set_title(f"Test {i+1}th neighbor")

plt.tight_layout(pad=0)
# print(sim_dev[:,0])

# %%
plt.rcParams["font.family"] = "serif"

fig, axs = plt.subplots(4, 2, figsize=(3.5, 7), sharex=True, sharey=True)

EMBEDDING2NAME = {
    "src": "Source",
    "mt": "Translation",
    "srcAmt": "Source + Translation",
    "srcCmt": "Source & Translation",
}

for i, EMBEDDING in enumerate(["src", "mt", "srcAmt", "srcCmt"]):
    sim_dev = np.load(f"../computed/sim_dev_retrieval_minilm_{THRESHOLD}_{EMBEDDING}.npy", allow_pickle=True)
    sim_test = np.load(f"../computed/sim_test_retrieval_minilm_{THRESHOLD}_{EMBEDDING}.npy", allow_pickle=True)


    sim_dev[sim_dev == None] = 0
    sim_test[sim_test == None] = 0

    axs[i,0].hist(
        sim_dev[:,0],
        bins=np.linspace(0.0, 1.0, 100),
        density=True,
        color="#fe7662",
    )
    axs[i,1].hist(
        sim_test[:,0],
        bins=np.linspace(0.0, 1.0, 100),
        density=True,
        color="#fe7662",
    )
    axs[i,0].set_title("Train\n" + EMBEDDING2NAME[EMBEDDING], fontsize=9, pad=-15)
    axs[i,1].set_title("Test\n" + EMBEDDING2NAME[EMBEDDING], fontsize=9, pad=-15)

    axs[i,0].spines[['top', 'right']].set_visible(False)
    axs[i,1].spines[['top', 'right']].set_visible(False)

    axs[i,0].set_xlabel("IP similarity", fontsize=8)
    axs[i,1].set_xlabel("IP similarity", fontsize=8)

    axs[i,0].set_yticks([])
    axs[i,0].set_ylabel("Relative frequency")

plt.tight_layout(pad=0.1)
plt.subplots_adjust(wspace=0.15, hspace=0.3)
# TODO: manually fix labels
plt.savefig("../figures/30-sim_histogram.pdf")