import numpy as np
import matplotlib.pyplot as plt

import bandits_lab.algorithms as algs
import bandits_lab.bandit_definitions as bands
import sim_utilities as sim

"""
    Code for the experiments in the paper Adaptation to the range in K-armed bandits.
"""

np.random.seed(0)
K = 50
T = 100000
n_tests = 300

"""
    Definitions of the problems considered.
"""
scales = [0.01, 0.1, 10, 100]
ups = np.ones(K)
lows = np.zeros(K)
ups[0] += 0.2
lows[0] += 0.2
M = 0

band_list = [bands.UnifDBand(K, scale * lows, scale * ups) for scale in scales]
N_tests = [n_tests for _ in band_list]

"""
    List of algorithms, all are implemented in the bandits_lab package.
"""
r = 1.2
alg_list = [
    algs.UCB_a(K, sig=0.01 * r, label=r"UCB $\sigma = 0.01 \times 1.2$"),
    algs.UCB_a(K, sig=0.1 * r, label=r"UCB $\sigma = 0.1 \times 1.2 $"),
    algs.UCB_a(K, sig=1 * r, label=r"UCB $\sigma = 1 \times 1.2 $"),
    algs.UCB_a(K, sig=10 * r, label=r"UCB $\sigma = 10 \times 1.2 $"),
    algs.UCB_a(K, sig=100 * r, label=r"UCB $\sigma = 100 \times 1.2 $"),
    # algs.Exp3(K, M=M, label="Exp3 "),
    # algs.AdaHedgeExp3(K, M=M, label="AHB without extra exploration"),
    algs.AdaHedgeExp3ExtraExp(K, label="AHB with extra exploration"),
    algs.UCB_a(K, sig=0, label="FTL"),
    algs.RandomPlay(K, label="Random Play"),
    # algs.FastAdaFTRLTsallis(
    #     K, M=M, sym=False, proxy=False, label="Bandit AdaFTRL Tsallis"
    # ),
    # algs.FastFTRLTsallis(K, M=M, label=r"FTRL Tsallis $\eta_t=t^{-1/2}$"),
    algs.MaxUCB(K, sig_init=0, label="MaxUCB"),
]

data_dict = {
    "name": "Long Name",
    "short_name": "multi_scale_2_k50",
    "T": T,
    "N_tests": N_tests,
    "band_list": band_list,
    "alg_list": alg_list,
    "results": None,
    "scales": scales,
    "seed": 0,
    "folder": "data_saves/range_adaptation/",
}

print("T :", T)
mus_list = list((ups + lows) / 2)
print("mus_list : ", mus_list)
print("scales : ", scales)

sim.launch(data_dict, n_jobs=4, checkpoints=True)

print("Done")

# skips = []
# sim.plot_and_save(
#     data_dict,
#     save_figure=False,
#     skip_algs=skips,
#     log_scale=False,
#     show_vars=False,
#     clean=True,
#     rescale=True,
# )  # , t_slice=t_slice)
#
# plt.show()
