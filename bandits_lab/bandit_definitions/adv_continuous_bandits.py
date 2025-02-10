import numpy as np


class ContBandPoly:
    """
    Semi-abstract class for continuous bandit problems.
    with polynomial rewards

    ie. functions of the form a0 + a1 * x + ... + ad * x**d
    """

    def __init__(self, d=5):
        self.time = 0

        self.d = d

        self.current_function_params = np.zeros(d)

        self.played_arms = []
        self.observed_rewards = []
        self.all_function_params = []
        self.cum_function_params = np.zeros(d)

    def _update_reward_params(self):
        """
        Modifies self.cum_function_params
        """
        return

    def play_arm(self, a):
        self._update_reward_params()
        reward = np.dot(self.current_function_params, [a**i for i in range(self.d)])

        self.played_arms.append(a)
        self.observed_rewards.append(reward)

        self.all_function_params.append(self.current_function_params)
        self.cum_function_params += self.current_function_params

        self.time += 1
        return reward

    def compute_cum_reward(self, x):
        return np.dot(self.cum_function_params, [x**i for i in range(self.d)])

    def reset(self):
        self.time = 0

        self.cum_function_params = np.zeros(self.d)

        self.played_arms = []
        self.observed_rewards = []
        self.all_function_params = []


class SimpleContBandPoly(ContBandPoly):
    def _update_reward_params(self):
        if self.time % 2 == 0:
            a = -np.arange(self.d)
        else:
            a = np.arange(self.d, 0, -1)
        self.current_function_params = a
        return
