import numpy as np


def bad_random_step(n):
    n2str = str(n**2)
    start_index = int(len(str(n2str)) / 4)
    # print(l)
    # print(n2str)
    return int(n2str[start_index : start_index + 5])


class ContBand:
    """
    Semi-abstract class for continuous bandit problems.
    with polynomial rewards
    """

    def __init__(self):
        self.time = 0
        self.played_arms = []
        self.observed_rewards = []

    def play(self, a):
        reward = self.compute_current_reward(a)

        self.played_arms.append(a)
        self.observed_rewards.append(reward)
        self.time += 1

        return reward

    def compute_current_reward(self, x):
        pass

    def compute_tth_reward(self, x, t):
        pass

    def compute_cum_reward(self, x):
        pass

    def reset(self):
        self.time = 0

        self.cum_function_params = np.zeros(self.d)

        self.played_arms = []
        self.observed_rewards = []


class ContBandPeak(ContBand):
    """ """

    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
        self.locations = []
        self.current_location = 0.5

        self.rng = 1984

    def _update_reward_params(self):
        if self.rng % 2:
            self.current_location = 0.6
        else:
            self.current_location = 0.4
        self.rng = self.rng + 1  # bad_random_step(self.rng)

    def _phi(self, x, x0, alpha):
        return 0.8 - np.power(np.abs(x - x0), alpha)

    def compute_current_reward(self, x):
        x0 = self.current_location
        return self._phi(x, x0, self.alpha)

    def compute_tth_reward(self, x, t):
        x0 = self.locations[t]
        return self._phi(x, x0, self.alpha)

    def compute_cum_reward(self, x):
        return np.sum([self._phi(x, x0, self.alpha) for x0 in self.locations])

    def play(self, a):
        reward = super().play(a)
        self.locations.append(self.current_location)
        self._update_reward_params()
        return reward


class ContBandPoly(ContBand):
    """
    Semi-abstract class for continuous bandit problems.
    with polynomial rewards

    ie. functions of the form a0 + a1 * x + ... + ad * x**d
    """

    def __init__(self, degree=5):
        super().__init__()
        self.degree = degree
        self.current_function_params = np.zeros(degree)
        self.all_function_params = []
        self.cum_function_params = np.zeros(degree)

    def _update_reward_params(self):
        """
        Modifies self.cum_function_params
        """
        pass

    def play(self, a):
        reward = super().play(a)

        self.all_function_params.append(self.current_function_params)
        self.cum_function_params += self.current_function_params
        self._update_reward_params()

        return reward

    def compute_current_reward(self, x):
        return np.dot(self.current_function_params, [x**i for i in range(self.degree)])

    def compute_tth_reward(self, x, t):
        return np.dot(self.all_function_params[t], [x**i for i in range(self.degree)])

    def compute_cum_reward(self, x):
        return np.dot(self.cum_function_params, [x**i for i in range(self.degree)])

    def reset(self):
        super().reset()

        self.current_function_params = np.zeros(self.degree)
        self.all_function_params = []
        self.cum_function_params = np.zeros(self.degree)
        self.cum_function_params = np.zeros(self.degree)


class SimpleContBandPoly(ContBandPoly):
    def _update_reward_params(self):
        if self.time % 2 == 0:
            a = -np.arange(self.degree)
        else:
            a = np.arange(self.degree, 0, -1)
        self.current_function_params = a
        return
