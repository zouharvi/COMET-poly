# %%

import numpy as np

sim_dev = np.load("../computed/sim_dev_retrieval_minilm_07_src.npy", allow_pickle=True)
sim_test = np.load("../computed/sim_test_retrieval_minilm_07_src.npy", allow_pickle=True)

sim_dev[sim_dev == None] = 0
sim_test[sim_test == None] = 0

import matplotlib.pyplot as plt

fig, axs = plt.subplots(5, 2, figsize=(4, 8), sharex=True, sharey=True)

for i in range(5):
    axs[i,0].hist(sim_dev[:,i], bins=np.linspace(0.0, 1.0, 100))
    axs[i,1].hist(sim_test[:,i], bins=np.linspace(0.0, 1.0, 100))

    axs[i,0].set_title(f"Dev {i+1}th neighbor")
    axs[i,1].set_title(f"Test {i+1}th neighbor")

plt.tight_layout(pad=0)
# print(sim_dev[:,0])