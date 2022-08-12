import numpy as np

import bandits_lab.algorithms as algs
import bandits_lab.bandit_definitions as bands
import sim_utilities as sim

"""
    Code for the experiments in the paper Adaptation to the range in K-armed bandits.
"""

np.random.seed(0)
K = 10
T = 100000
n_tests = 300


"""
    Definitions of the problems considered.
"""
scales = [1e-2, 1, 1e2]
ups = np.ones(K)
lows = np.zeros(K)
ups[0] = 1.2
M = 0

means = (ups + lows) / 2
# Low variance,  i.e., sigma = 0.1
V, short_name = 0.01, "multi_scale_fin_trunc_gauss_lo_var"
variances = V * np.ones(K)
band_list = [
    bands.SymTruncatedGaussian(K, s * lows, s * ups, s * means, s * s * variances)
    for s in scales
]
N_tests = [n_tests for _ in band_list]

"""
    List of algorithms. These algorithms are implemented in the bandits_lab package.
"""
r = 1.2
alg_list = [
    algs.UCB_a(K, sig=1e-2 * np.sqrt(4 * V), label=r"UCB $\sigma = 0.01 \times 1.2$"),
    algs.UCB_a(K, sig=1e0 * np.sqrt(4 * V), label=r"UCB $\sigma = 1 \times 1.2$"),
    algs.UCB_a(K, sig=1e2 * np.sqrt(4 * V), label=r"UCB $\sigma = 100 \times 1.2 $"),
    # algs.AdaHedgeExp3(K, M=M, label="AHB without extra exploration M=0"),
    # algs.AdaHedgeExp3(K, M=1.1, label="AHB without extra exploration M=1.2"),
    algs.MaxUCB(K, sig_init=0, label="Range estimating UCB"),
    algs.AdaHedgeExp3ExtraExp(K, label="AHB with extra exploration"),
    algs.UCB_a(K, sig=0, label="FTL"),
    # algs.EpsGreedy(K, c=5, d=1e-2 * 0.1, label=r"$\varepsilon$-greedy, $d = 0.01$"),
    algs.EpsGreedy(K, c=5, d=(0.1 / 1.2), label=r"$\varepsilon$-greedy, $d=1$"),
    # algs.EpsGreedy(K, c=5, d=1e2 * 0.1, label=r"$\varepsilon$-greedy, $d=100$"),
    algs.RandomPlay(K, label="Random Play"),
    # algs.FastAdaFTRLTsallis(
    #     K, M=M, sym=False, proxy=False, label="Bandit AdaFTRL Tsallis"
    # ),
    # algs.FastFTRLTsallis(K, M=M, label=r"FTRL Tsallis $\eta_t=t^{-1/2}$"),
]


data_dict = {
    "name": "Long Name",
    "short_name": short_name,
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
print("means list : ", means)
print("scales : ", scales)


sim.launch(data_dict, n_jobs=4, checkpoints=True)


"""
Round 2
"""


# High-variance, i.e., sigma = 0.5
V, short_name = 0.25, "multi_scale_fin_trunc_gauss_hi_var"
variances = V * np.ones(K)
band_list = [
    bands.SymTruncatedGaussian(K, s * lows, s * ups, s * means, s * s * variances)
    for s in scales
]
N_tests = [n_tests for _ in band_list]

"""
    List of algorithms. These algorithms are implemented in the bandits_lab package.
"""
r = 1.2  # / 4 to have UCB with the 2 downstairs
alg_list = [
    algs.UCB_a(K, sig=1e-2 * np.sqrt(4 * V), label=r"UCB $\sigma = 0.01 \times 1.2$"),
    algs.UCB_a(K, sig=1e0 * np.sqrt(4 * V), label=r"UCB $\sigma = 1 \times 1.2$"),
    algs.UCB_a(K, sig=1e2 * np.sqrt(4 * V), label=r"UCB $\sigma = 100 \times 1.2 $"),
    # algs.Exp3(K, M=1.2, label="Exp3 M=1.2"),
    # algs.AdaHedgeExp3(K, M=M, label="AHB without extra exploration M=0"),
    # algs.AdaHedgeExp3(K, M=1.1, label="AHB without extra exploration M=1.2"),
    algs.MaxUCB(K, sig_init=0, label="Range estimating UCB"),
    algs.AdaHedgeExp3ExtraExp(K, label="AHB with extra exploration"),
    algs.EpsGreedy(K, c=T, d=1, label=r"FTL"),
    algs.EpsGreedy(K, c=5, d=1e0 * 0.1, label=r"$\varepsilon$-greedy, $d=1$"),
    algs.RandomPlay(K, label="Random Play"),
    # algs.FastAdaFTRLTsallis(
    #     K, M=M, sym=False, proxy=False, label="Bandit AdaFTRL Tsallis"
    # ),
    # algs.FastFTRLTsallis(K, M=M, label=r"FTRL Tsallis $\eta_t=t^{-1/2}$"),
]

data_dict = {
    "name": "Long Name",
    "short_name": short_name,
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
print("means list : ", means)
print("scales : ", scales)

sim.launch(data_dict, n_jobs=2, checkpoints=True)


print("Done")
