import numpy as np

import bandits_lab.algorithms as algs
import bandits_lab.bandit_definitions as bands
import sim_utilities as sim

np.random.seed(10)
K = 2

T = 10000

means = np.array([0.2, 0.1])
mus = 0.5 + np.array([0.2, 0.1])
variances = 0.01 * np.ones(K)

band_list = [bands.BernoulliBand(K, means),
        bands.TruncatedGaussian(K, mus, variances),
        ]

N_tests = [500, 500]

alg_list = [algs.KLUCB(K, label="KL-UCB"),
        algs.klUCBplusplus(K, label="kl-UCB++"),
        algs.MOSS_a(K, sig=1/2, label="MOSS"),
        algs.KLUCBswitch(K, label="Switch"),
        algs.KLUCBswitchPro(K, label="SwitchPro"),
        ]

data_dict={
         'name':'KL-UCB',
         'short_name':'klucb2_anytime',
         'T':T,
         'N_tests':N_tests,
         'band_list':band_list,
         'alg_list':alg_list,
         'folder': 'data_saves/kl_ucb/',
         'results':None,
     }

sim.launch(data_dict, n_jobs=4, checkpoints=True)
