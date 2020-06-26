import numpy as np
import matplotlib.pyplot as plt
import pickle
import time

from bandit_definitions import *
from algorithms import *
from sim_utilities import *

np.random.seed(0)

K = 30
T = 100000

ds = 0.3*np.ones(K)
ds[0] = 0.1
M = 0
scales = [.01, .1, 1, 10, 100]
mus_list = [M - scale*(1/2 + ds) for scale in scales]

band_list = [UnifDBand(K, mus_list[i], M, scale) for i, scale in enumerate(scales)]
N_tests = [100 for _ in band_list]

alg_list = [
    UCB(K, sig=.01, label="UCB \sigma = "+str(.01)),
    UCB(K, sig=.1, label="UCB \sigma = "+str(.1)),
    UCB(K, sig=1, label="UCB \sigma = 1"),
    UCB(K, sig=10, label="UCB \sigma = 10"),
    UCB(K, sig=100, label="UCB \sigma = 100"),
    Exp3(K , M=M, label="Vanilla Exp3"),
    AdaHedgeExp3(K, M=M, label="True AdaExp3"),
    UCB(K, sig=0, label="FTL"),
    RandomPlay(K, label="random play"),
    FastAdaFTRLTsallis(K, M=M, sym=False, proxy=True, speed=.5, label="FastAdaFTRL Tsallis prox"),
    FastFTRLTsallis(K, M=M, speed=.5, label="Fast FTRL Tsallis eta=1/sqrt(t)"),
    ]

#For FastFTRLTsallis convergence does not really depend on speed: always take .5 and will converge is scale < 10
#For FastAdaFTRLTsallis best speed is .5


data_dict={
    'name':'Long Name',
    'short_name':'multi_scale2',
    'T':T,
    'N_tests':N_tests,
    'band_list':band_list,
    'alg_list':alg_list,
    'results':None,
    'scales':scales,
    'seed':0,
    'folder':'figures/testing/',
}

print("T", T)
print('mus_list', mus_list)
print("scales", scales)

launch(data_dict, fair_reg=False, n_jobs=4, checkpoints=True)

print("Done")

#t_slice = range(int(1e3), int(1e4))
skips = []
plot_and_save(data_dict,
                save_figure=False,
                skip_algs=skips,
                log_scale=False,
                show_vars=False,
                rescale=True)#, t_slice=t_slice)

plt.show()
