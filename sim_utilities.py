import numpy as np
import os
import pickle
import time
import matplotlib.pyplot as plt

from bandit_definitions import *
from algorithms import *
from copy import deepcopy

from joblib import Parallel, delayed

################# running the simuations from a data_dict:
# data_dict={
#         'name':'Long Name',
#         'short_name':'short_name',
#         'T':T,
#         'N_tests':N_tests,
#         'band_list':band_list,
#         'alg_list':alg_list,
#         'results':None,
#     }


############################# Playing algorithms

def playTtimes(bandit, alg, T):
    for t in range(T):
        alg.play_once(bandit)
    return

def one_regret(a, b, T, *, fair_reg):
    alg = deepcopy(a)
    bandit = deepcopy(b)
    playTtimes(bandit, alg, T)
    if fair_reg:
        return bandit.cum_fair_regret
    else:
        return bandit.cum_regret

def n_regret(alg, bandit, T, N_test=100, verb=False, n_jobs=1, *, fair_reg):
    return Parallel(n_jobs=n_jobs)(delayed(one_regret)(alg, bandit, T, fair_reg=fair_reg) for _ in range(N_test))


############################# Saving utilities:
def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1
    while os.path.exists(path):
        path = filename + "_" + str(counter) + extension
        counter += 1
    return path

def save_data_dict(data_dict, uniquify=False):
    folder=data_dict['folder']
    if uniquify:
        path = uniquify(folder+data_dict['short_name']+'.pkl')
    else:
        path = folder+data_dict['short_name']+'.pkl'
    with open(path, 'wb') as f:
        pickle.dump(data_dict, f)

def load_data_dict(path):
    with open(path, 'rb') as f:
        data_dict = pickle.load(f)
    return data_dict

def launch(data_dict, verb=False, n_jobs=1, checkpoints=True, *, fair_reg):
    if 'seed' in data_dict.keys():
        np.random.seed(data_dict['seed'])
    T, band_list = data_dict['T'], data_dict['band_list']
    alg_list, N_tests = data_dict['alg_list'], data_dict['N_tests']
    n_algs = len(alg_list)
    results = []
    for i, band in enumerate(band_list):
        time_comp = []
        results.append([])
        for j, alg in enumerate(alg_list):
            t0  = time.time()
            N_test = N_tests[i]
            temp = np.array(n_regret(alg, band, T, N_test=N_test, verb=verb, n_jobs=n_jobs, fair_reg=fair_reg))
            time_comp.append((time.time() - t0))
            print(alg.label, ' took ', time_comp[j],' total, i.e., ', time_comp[j]/N_test, ' per run')
            mean_reg = np.mean(temp, axis=0)
            var_reg = np.var(temp, axis=0)
            results[-1].append((mean_reg, var_reg))
            data_dict['results'] = results
            if checkpoints:
                print('saved')
                save_data_dict(data_dict, uniquify=False)


def plot_and_save(data_dict, save_data=False, skip_algs=[], log_scale=True, show_vars=True, **kwargs):
    """ Used to hard save the data """
    colors = plt.get_cmap('tab20').colors
    T = data_dict['T']
    if 't_slice' in kwargs:
        t_slice = kwargs['t_slice']
    else:
        t_slice = range(T)
    nplots = len(data_dict['band_list'])
    fig, axes = plt.subplots(nrows=1, ncols=nplots, figsize=(16, 4), sharey='all')
    for i, _ in enumerate(data_dict['band_list']):
        if nplots >= 2:#if nplots >= 2: # weird : axes[i] does not work when there is only 1 subplot
            ax = axes[i]
        else:
            ax = axes
        for j, alg in enumerate(data_dict['alg_list']):
            if j in skip_algs:
                continue
            mean_reg, var_reg = data_dict['results'][i][j]
            if ('rescale' in kwargs.keys()) & kwargs['rescale']:
                mean_reg = np.array(mean_reg) / data_dict['scales'][i]
                var_reg = np.array(var_reg) / np.square(data_dict['scales'][i])
            if log_scale:
                ax.set_xscale("log")#, nonposx='clip')
            ax.plot(t_slice, mean_reg[t_slice], label=str(j)+": "+alg.label, color=colors[j])
            if show_vars:
                ax.plot(t_slice, mean_reg[t_slice]+ np.sqrt(var_reg[t_slice]), '--', alpha=0.3, color=colors[j])
                ax.plot(t_slice, mean_reg[t_slice]- np.sqrt(var_reg[t_slice]), '--', alpha=0.3, color=colors[j])
        ax.legend()

    if save_data:
        plt.tight_layout()
        save_data_dict(data_dict)
        path = uniquify('figures/'+data_dict['short_name']+'.pdf')
        plt.savefig(path, format='pdf')
