import numpy as np
import os
import pickle
import time
import matplotlib.pyplot as plt

# import bandits_lab.bandit_definitions as bands
import bandits_lab.algorithms as algs

from copy import deepcopy

from joblib import Parallel, delayed

# running the simuations from a data_dict:
# data_dict={
#         'name':'Long Name',
#         'short_name':'short_name',
#         'T':T,
#         'N_tests':N_tests,
#         'band_list':band_list,
#         'alg_list':alg_list,
#         'results':None,
#     }


# Playing algorithms


def one_regret(a, b, T):
    alg = deepcopy(a)
    bandit = deepcopy(b)
    converged = True
    try:
        alg.play_T_times(bandit, T)
        return bandit.cum_regret, converged
    except algs.ConvergenceError:
        converged = False
        return None, converged


def n_regret(alg, bandit, T, N_test=100, verb=False, n_jobs=1):
    reg_list = Parallel(n_jobs=n_jobs)(
        delayed(one_regret)(alg, bandit, T) for _ in range(N_test)
    )
    n_regret_list = []
    all_converged = True
    for regret, converged in reg_list:
        if not (converged):
            all_converged = False
            continue
        else:
            n_regret_list.append(regret)
    if all_converged:
        return np.array(n_regret_list), True
    else:
        return None, False


# Saving utilities:
def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1
    while os.path.exists(path):
        path = filename + "_" + str(counter) + extension
        counter += 1
    return path


def save_data_dict(data_dict, uniquify=False):
    folder = data_dict["folder"]
    if uniquify:
        path = uniquify(folder + data_dict["short_name"] + ".pkl")
    else:
        path = folder + data_dict["short_name"] + ".pkl"
    with open(path, "wb") as f:
        pickle.dump(data_dict, f)


def load_data_dict(path):
    with open(path, "rb") as f:
        data_dict = pickle.load(f)
    return data_dict


def launch(data_dict, verb=False, n_jobs=1, checkpoints=True):
    if "seed" in data_dict.keys():
        np.random.seed(data_dict["seed"])
    T, band_list = data_dict["T"], data_dict["band_list"]
    alg_list, N_tests = data_dict["alg_list"], data_dict["N_tests"]
    results = []
    time_comp = []
    ended = []
    for i, band in enumerate(band_list):
        results.append([])
        time_comp.append([])
        ended.append([])
        N_test = N_tests[i]
        for j, alg in enumerate(alg_list):
            t0 = time.time()
            n_regret_array, all_converged = n_regret(
                alg, band, T, N_test=N_test, verb=verb, n_jobs=n_jobs
            )
            if all_converged:
                time_taken = time.time() - t0
                time_comp[-1].append(time_taken)
                ended[-1].append(True)
                print(
                    f"{alg.label} took {time_taken} total, i.e.,"
                    + f" {time_taken / N_test} per run",
                )
                mean_reg = np.mean(n_regret_array, axis=0)
                var_reg = np.var(n_regret_array, axis=0)
                results[-1].append((mean_reg, var_reg))
                data_dict["time_comp"] = time_comp
                data_dict["results"] = results
                data_dict["ended"] = ended
                if checkpoints:
                    print("saved")
                    save_data_dict(data_dict, uniquify=False)
            else:
                time_comp[-1].append(None)
                results[-1].append((None, None))
                ended[-1].append(False)
                print(alg.label, " failed to converge")
                if checkpoints:
                    print("saved")
                    save_data_dict(data_dict, uniquify=False)
                continue


def plot_and_save(
    data_dict,
    save_figure=False,
    skip_algs=[],
    log_scale=True,
    show_vars=True,
    clean=False,
    **kwargs,
):
    """ Tailored to the range adaptation experiments """
    colors = plt.get_cmap("tab20").colors
    T = data_dict["T"]
    if "t_slice" in kwargs:
        t_slice = kwargs["t_slice"]
    else:
        t_slice = range(T)
    nplots = len(data_dict["band_list"])
    fig, axes = plt.subplots(nrows=1, ncols=nplots, figsize=(16, 4), sharey="all")
    for i, _ in enumerate(data_dict["band_list"]):
        if nplots >= 2:  # axes[i] does not work when there is only 1 subplot
            ax = axes[i]
        else:
            ax = axes
        for j, alg in enumerate(data_dict["alg_list"]):
            if j in skip_algs or not (data_dict["ended"][i][j]):
                continue
            mean_reg, var_reg = data_dict["results"][i][j]
            if "rescale" in kwargs.keys():
                if kwargs["rescale"]:
                    mean_reg = np.array(mean_reg) / data_dict["scales"][i]
                    var_reg = np.array(var_reg) / np.square(data_dict["scales"][i])
            if log_scale:
                ax.set_xscale("log")  # , nonposx='clip')
            if clean:
                alg_label = alg.label
            else:
                alg_label = str(j) + ": " + alg.label
            ax.plot(
                t_slice, mean_reg[t_slice], label=alg_label, color=colors[j],
            )
            if show_vars:
                sig = np.sqrt(var_reg[t_slice] / data_dict["N_tests"][i])
                ax.fill_between(
                    t_slice,
                    mean_reg[t_slice] + 2 * sig,
                    mean_reg[t_slice] - 2 * sig,
                    alpha=0.3,
                    color=colors[j],
                )
        if i == 0:
            ax.legend()

    if save_figure:
        plt.tight_layout()
        # save_data_dict(data_dict)
        path = uniquify(data_dict["short_name"] + ".pdf")
        plt.savefig(path, format="pdf")
