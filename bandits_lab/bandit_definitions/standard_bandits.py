import numpy as np


class DBand:
    """ K-arm stochastic bandit """

    def __init__(self, K, mus, noise="gaussian"):
        self.K = K
        self.mus = mus
        self.m = np.max(mus)  # max mean value
        self.noise = noise  # String describing the noise distributions
        self.time = 0

        self.played_arms = []
        self.observed_rewards = []
        self.point_regret = []
        self.cum_regret = [0]

    def _compute_reward(self, a):
        pass

    def play_arm(self, a):
        reward = self._compute_reward(a)
        self.played_arms.append(a)
        self.observed_rewards.append(reward)
        self.point_regret.append(self.m - self.mus[a])

        self.time += 1
        last_r = self.cum_regret[-1]
        self.cum_regret.append(last_r + self.m - self.mus[a])

        return reward

    def reset(self):
        self.time = 0

        self.played_arms = []
        self.observed_rewards = []
        self.point_regret = []
        self.cum_regret = []


class GaussBand(DBand):
    def __init__(self, K, mus, variances):
        super().__init__(K, mus, noise="gaussian")
        self.variances = variances
        self.sigma = np.sqrt(self.variances)

    def _compute_reward(self, a):
        return np.random.normal(self.mus[a], self.sigma[a])


class BernoulliBand(DBand):
    def __init__(self, K, mus):
        assert all((self.mus <= 1) and (self.mus >= 0))
        super().__init__(K, mus, noise="bernoulli")

    def _compute_reward(self, a):
        return 1.0 * (np.random.rand() < self.mus[a])


class UnifDBand(DBand):
    """ uniform bandits defined via the supports of each arm """

    def __init__(self, K, lows, ups):
        mus = [(low + up) / 2 for low, up in zip(lows, ups)]
        super().__init__(K, mus, noise="unif")
        self.lows = lows
        self.ups = ups

    def _compute_reward(self, a):
        return self.lows[a] + np.random.rand() * (self.ups[a] - self.lows[a])


class SymTruncatedGaussian(DBand):
    """
    Symmetric Truncated gaussians: specify lower, upper, means, variances
    symmetric means that the means have to be ( l + u )/ 2.
    """

    def __init__(self, K, lows, ups, means, variances):
        super().__init__(K, means, noise="TruncGauss")
        assert np.all(means == (lows + ups) / 2)
        self.lows = lows
        self.ups = ups
        self.sigma = np.sqrt(variances)

    def _compute_reward(self, a):
        l, u, mu, sigma = self.lows[a], self.ups[a], self.mus[a], self.sigma[a]
        return max(l, min(u, np.random.normal(mu, sigma)))


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
