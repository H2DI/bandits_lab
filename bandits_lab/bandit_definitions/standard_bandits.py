import numpy as np


class DBand:
    """ K-arm stochastic bandit """

    def __init__(self, K, mus, noise="gaussian", noise_sig=0.25):
        self.K = K
        self.mus = mus
        self.m = np.max(mus)  # max mean value
        self.noise = noise  # either "gaussian" or "bernoulli"
        self.noise_sig = noise_sig
        self.time = 0

        if self.noise == "bernoulli":
            assert (self.mus <= 1).all() and (self.mus >= 0).all()

        self.played_arms = []
        self.observed_rewards = []
        self.point_regret = []
        self.cum_regret = []

    def _compute_reward(self, a):
        if self.noise == "gaussian":
            return self.mus[a] + np.random.normal(0, self.noise_sig)
        elif self.noise == "bernoulli":
            return np.random.rand() <= self.mus[a]

    def play_arm(self, a):
        reward = self._compute_reward(a)
        self.played_arms.append(a)
        self.observed_rewards.append(reward)
        self.point_regret.append(self.m - self.mus[a])

        self.time += 1
        if self.cum_regret:
            last_r = self.cum_regret[-1]
            self.cum_regret.append(last_r + self.m - self.mus[a])
        else:
            self.cum_regret.append(self.m - self.mus[a])
        return reward

    def reset(self):
        self.time = 0

        self.played_arms = []
        self.observed_rewards = []
        self.point_regret = []
        self.cum_regret = []


class UnifDBand(DBand):
    """ uniform bandits defined via the supports of each arm """

    def __init__(self, K, lows, ups):
        mus = [(low + up) / 2 for low, up in zip(lows, ups)]
        super().__init__(K, mus, noise="unif")
        self.lows = lows
        self.ups = ups

    def _compute_reward(self, a):
        return self.lows[a] + np.random.rand() * (self.ups[a] - self.lows[a])


class UnifDBandBand2(DBand):
    """obsolete / kept in case I forgot something"""

    def __init__(self, K, mus, M, r_range):
        super().__init__(K, mus, noise="unif")
        self.range = r_range
        assert all(self.mus < M - self.range / 2)

    def _compute_reward(self, a):  # returns a couple (mean, actual_reward)
        noise = self.range * (np.random.rand() - 1 / 2)
        return self.mus[a] + noise


class AdvObliviousBand:
    """
        Adversarial Oblivious Bandit. Never tested.

        The reward_gen attribute is a function that takes time as an arguments and
        generates a reward vector (np.Array)
    """

    def __init__(self, K, reward_gen):
        self.K = K
        self.reward_gen = reward_gen

        self.time = 0
        self.played_arms = []
        self.observed_rewards = []
        self.cumulative_received_reward = 0
        self.all_comparator_rewards = np.zeros(self.K)  # unobserved by the algorithm
        self.cum_regret = []

    def play_arm(self, a):
        self.time += 1
        self.played_arms.append(a)
        rewards = self.reward_gen(self.time)
        self.observed_rewards.append(rewards[a])
        self.all_comparator_rewards += rewards
        self.cumulative_received_reward += rewards[a]
        self.cum_regret.append(
            np.max(self.all_comparator_rewards) - self.cumulative_received_reward
        )
        return rewards[a]

    def reset(self):
        self.time = 0
        self.played_arms = []
        self.observed_rewards = []
        self.cumulative_received_reward = 0
        self.all_comparator_rewards = np.zeros(self.K)
        self.cum_regret = []
