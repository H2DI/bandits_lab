import numpy as np
import matplotlib.pyplot as plt
import pickle
import time

from bandit_definitions import *
from algorithms import *
from sim_utilities import *

np.random.seed(0)


K = 5
T = 10000

ds = 0.5*np.ones(K)
ds[0] = 0.1
M = 1
scales = [1]
mus_list = [M - scale*(1/2 + ds) for scale in scales]

band_list = [UnifDBand(K, mus_list[i], M, scale) for i, scale in enumerate(scales)]
N_tests = [2 for _ in band_list]

alg_list = [
    # UCB(K, sig=0.1, label="UCB \sigma = "+str(0.1)),
    # UCB(K, sig=1, label="UCB \sigma = 1"),
    # UCB(K, sig=200, label="UCB \sigma = 200"),
    Exp3(K , M=M, label="Vanilla Exp3"),
    # AdaHedgeExp3(K, M=M, label="True AdaExp3"),
    # FastAdaFTRLTsallis(K, M=M, sym=False, proxy=True, label="FastAdaFTRL Tsallis prox"),
    FastFTRLTsallis(K, M=M, label="Fast FTRL Tsallis eta=1/sqrt(T)"),
    # UCB(K, sig=0, label="FTL"),
    RandomPlay(K, label="random play"),
    ]


data_dict={
    'name':'Long Name',
    'short_name':'test102',
    'T':T,
    'N_tests':N_tests,
    'band_list':band_list,
    'alg_list':alg_list,
    'results':None,
    'scales':scales,
    'seed':0,
    'folder':'figures/testing/',
}

launch(data_dict, fair_reg=False, n_jobs=4, checkpoints=True)

print("Done")

#t_slice = range(int(1e3), int(1e4))
skips = []
plot_and_save(data_dict, save_data=False, skip_algs=skips, log_scale=False, show_vars=False,
              rescale=True)#, t_slice=t_slice)

plt.show()
