import numpy as np

import bandits_lab.algorithms as algs
import bandits_lab.bandit_definitions as bands
import sim_utilities as sim

np.random.seed(10)
T = 20000
n_tests = 75

"""
    Definition of the problems considered
        - the probability set considered is a triangle,
        - we build a family of  bandit problems, defined by their mean vectors.
        For the last two mean vectors, the optimal (mixed) action in on the border
        of the simplex, whereas for the fist two, the optimal action is in the
        interior of the simplex. We know that the regret has to grow at least
        logarithmically in the latter case.

        We conjecture that diversity-preserving UCB yields bounded regret when
        the optimal action is on the border.
"""

K = 3

low = 0.1
lows = [0] + [low for _ in range(K - 1)]
constraints_list = [
    (lows[i], 1, np.array([1 * (j == i) for j in range(K)])) for i in range(K)
]
setP = bands.PolytopeConstraints(K, constraints_list)

delta = np.array([0.05, 0, -0.05])
mus = np.array([1 / 2, 1 / 3, 1 / 2])
mus_list = [
    mus - 2 * delta,
    mus - delta,
    mus,
    mus + delta,
    mus + 2 * delta,
]

min_reward, max_reward = setP.argmax_dot(-mus), setP.argmax_dot(mus)
print(max_reward)
delta_max = max_reward.fun - (-min_reward.fun)
# print("Un point qui vérifie les contraintes : ", C.feasible)
print("Maximum gap for this problem : ", delta_max)

band_list = [bands.DivPBand(K, mus, setP) for mus in mus_list]

"""
    Definition of the algorithms.

    We are only interested in (diversity-preserving) UCB here. We also run
    follow-the-leader for control.
"""


eps = 1
alg_list = [  # DivPUCB(K, C, label='DivP kl-UCB'),
    algs.DivPUCB(K, setP, sig=1, label="DivP UCB"),
    # L1OFUL(K, C, label="L1OFUL", delta=1/T),
    # DivPEpsGreedy(
    #    K, setP, label=r"$\epsilon$-greedy, $\epsilon =$" + str(eps), epsilon=eps
    # ),
    # algs.DivPEpsGreedy(K, setP, label="Follow-the-leader", epsilon=0),
]

N_tests = [n_tests for _ in band_list]

data_dict = {
    "name": "Transition from logarithmic to bounded regret gaussian noise",
    "short_name": "log_or_bounded_regret",
    "T": T,
    "N_tests": N_tests,
    "band_list": band_list,
    "alg_list": alg_list,
    "results": None,
    "folder": "data_saves/diversity/",
}

# "data_saves/diversity/log_or_bounded_regret"

sim.launch(data_dict, checkpoints=True, n_jobs=4)

skips = []
sim.plot_and_save(
    data_dict, save_figure=False, skip_algs=skips, log_scale=True, show_vars=True,
)

print("done")
