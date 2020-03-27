import numpy as np
import os
import pickle
import time
import matplotlib.pyplot as plt

from bandit_definitions import *
from algorithms import *

from joblib import Parallel, delayed

################# running the simuations from a data_dict:
# data_dict={
#         'name':'Long Name',
#         'short_name':'short_name',
#         'T':T,
#         'Ntest':Ntest,
#         'band_list':band_list,
#         'alg_list':alg_list,
#         'results':None,
#     }

def playTtimes(bandit, alg, T):
    for t in range(T):
        alg.play_once(bandit)
    return

def one_regret(alg, bandit, T, *, fair_reg):
    alg.reset()
    bandit.reset()
    playTtimes(bandit, alg, T)
    if fair_reg:
        return bandit.cum_fair_regret
    else:
        return bandit.cum_regret

# def average_regret(alg, bandit, T, N_test=100, verb=False, *, fair_reg, n_jobs=1):
#     regrets = []
#     for i in range(N_test):
#         # if verb:
#         #     if not(i % 5):
#         #         print("Run number ", i, " / ", N_test)
#         regrets.append(one_regret)
#     return regrets

def average_regret(alg, bandit, T, N_test=100, verb=False, n_jobs=1, *, fair_reg):
    return Parallel(n_jobs=n_jobs)(delayed(one_regret)(alg, bandit, T, fair_reg=fair_reg) for _ in range(N_test))

def launch(data_dict, verb=False, n_jobs=1,*, fair_reg):
    if 'seed' in data_dict.keys():
        np.random.seed(data_dict['seed'])
    T, band_list = data_dict['T'], data_dict['band_list']
    alg_list, Ntest = data_dict['alg_list'], data_dict['Ntest']
    results = []
    for band in band_list:
        mean_regs = np.zeros((len(alg_list), T))
        var_regs = np.zeros((len(alg_list), T))
        time_comp = np.zeros(len(alg_list))
        for i, alg in enumerate(alg_list):
            t0  = time.time()
            temp = np.array(average_regret(alg, band, T, N_test=Ntest(T), verb=verb, n_jobs=n_jobs, fair_reg=fair_reg))
            time_comp[i] = (time.time() - t0)
            print(alg.label, ' took ', time_comp[i],' total, i.e., ', time_comp[i]/Ntest(T), ' per run')
            mean_regs[i] = np.mean(temp, axis=0)
            var_regs[i] = np.var(temp, axis=0)

        results.append((mean_regs, var_regs))
    data_dict['results'] = results


############################# Saving utilities:
def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1
    while os.path.exists(path):
        path = filename + "_" + str(counter) + extension
        counter += 1
    return path

def save_data_dict(data_dict, folder):
    path = uniquify(folder+data_dict['short_name']+'.pkl')
    with open(path, 'wb') as f:
        pickle.dump(data_dict, f)

def plot_and_save(data_dict, save_data=False, skip_algs=[], log_scale=True, show_vars=True, **kwargs):
    colors = plt.get_cmap('tab10').colors
    T = data_dict['T']
    if 't_slice' in kwargs:
        t_slice = kwargs['t_slice']
    else:
        t_slice = range(T)
    nplots = len(data_dict['band_list'])
    fig, axes = plt.subplots(nrows=1, ncols=nplots, figsize=(16, 4), sharey='all')
    if nplots >= 2: # weird : axes[j] does not work when there is only 1 subplot
        for j, _ in enumerate(data_dict['band_list']):
            mean_regs, var_regs = data_dict['results'][j]
            if ('rescale' in kwargs.keys()) & kwargs['rescale']:
                mean_regs = mean_regs / data_dict['scales'][j]
                var_regs = var_regs / np.square(data_dict['scales'][j])
            for i, alg in enumerate(data_dict['alg_list']):
                if i in skip_algs:
                    continue
                if log_scale:
                    axes[j].set_xscale("log")#, nonposx='clip')
                axes[j].plot(mean_regs[i, t_slice], label=str(i)+": "+alg.label, color=colors[i])
                if show_vars:
                    axes[j].plot(mean_regs[i, t_slice]+ np.sqrt(var_regs[i]), '--', alpha=0.3, color=colors[i])
                    axes[j].plot(mean_regs[i, t_slice]- np.sqrt(var_regs[i]), '--', alpha=0.3, color=colors[i])
            axes[j].legend()
    else:
        mean_regs, var_regs = data_dict['results'][0]
        if ('rescale' in kwargs.keys()) & kwargs['rescale']:
            mean_regs /= data_dict['scales'][0]
            var_regs /= data_dict['scales'][0]
        for i, alg in enumerate(data_dict['alg_list']):
            if i in skip_algs:
                continue
            if log_scale:
                axes.set_xscale("log")#, nonposx='clip')
            axes.plot(mean_regs[i, t_slice], label=str(i)+": "+alg.label, color=colors[i])
            if show_vars:
                axes.plot(mean_regs[i, t_slice]+ np.sqrt(var_regs[i]), '--', alpha=0.3, color=colors[i])
                axes.plot(mean_regs[i, t_slice]- np.sqrt(var_regs[i]), '--', alpha=0.3, color=colors[i])
        axes.legend()

    if save_data:
        plt.tight_layout()
        save_data_dict(data_dict, 'figures/')
        path = uniquify('figures/'+data_dict['short_name']+'.pdf')
        plt.savefig(path, format='pdf')
