import numpy as np


class ContBand:
    """
        Semi-abstract class for continuous bandit problems. To use this you need to:
        - implement the self._compute_mean_reward method,
        - compute the maximal value of the mean reward and set self.m .
    """

    def __init__(self, noise="gaussian", noise_sig=0.25):
        self.m = None
        self.time = 0

        self.noise = noise
        self.nois_sig = noise_sig

        self.played_arms = []
        self.observed_rewards = []
        self.point_regret = []
        self.cum_regret = []

    def _compute_mean_reward(self, a):
        pass

    def _compute_reward(self, a):
        mean_reward = self._compute_mean_reward(a)
        if self.noise == "gaussian":
            reward = mean_reward + np.random.normal(0, 0.25)
        elif self.noise == "bernoulli":
            reward = np.random.random() < mean_reward
        return mean_reward, reward

    def play_arm(self, a):
        mean_reward, reward = self._compute_reward(a)
        self.played_arms.append(a)
        self.observed_rewards.append(reward)
        self.point_regret.append(self.m - mean_reward)

        self.time += 1
        if self.cum_regret:
            a = self.cum_regret[-1]
            self.cum_regret.append(a + self.m - mean_reward)
        else:
            self.cum_regret.append(self.m - mean_reward)
        return reward

    def reset(self):
        self.time = 0

        self.played_arms = []
        self.observed_rewards = []
        self.point_regret = []
        self.cum_regret = []


################################################################################


class PeakBandit(ContBand):
    r""" simplest $L, \alpha$ - Hölder function """

    def __init__(self, m, x, L, alpha):
        super().__init__()
        self.m = m
        self.x = x
        self.L = L
        self.alpha = alpha

    def _compute_mean_reward(self, y):
        return max(self.m - self.L * np.power(np.abs(self.x - y), self.alpha), 0.0)


class SmallPeakBandit(ContBand):
    """ Define a bandit problem with a small peak and flat everywhere else """

    def __init__(self, m, x, L, alpha, T, height=-1):
        super().__init__()
        self.m = m
        self.x = x
        self.L = L
        self.alpha = alpha
        self.T = T

        if height == -1:
            self.delta = np.power(L, 1 / (2 * alpha + 1)) * np.power(
                T, -alpha / (2 * alpha + 1)
            )
            self.height = self.m - self.delta
        else:
            self.height = height

    def _compute_mean_reward(self, y):
        return max(
            self.m - self.L * np.power(np.abs(self.x - y), self.alpha), self.height
        )


class GarlandBand(ContBand):
    # Used in experiments
    def __init__(self):
        super().__init__()
        self.m = 0.9968024935005756

    def _compute_mean_reward(self, y):
        return y * (1 - y) * (4 - np.sqrt(np.abs(np.sin(60 * y))))


class ParabolePic(ContBand):
    # Used in experiments
    def __init__(self, a=0.05, b=3.6):
        super().__init__()
        self.a = a
        self.b = b
        self.m = max(b / 4, 1.0)

    def _compute_mean_reward(self, y):
        a = self.a
        return max(self.b * y * (1 - y), 1 - 1 / a * np.abs(y - a))


class WeirdSinus(ContBand):
    # Used in experiments
    def __init__(self):
        super().__init__()
        self.m = 0.9755991424707509

    def _compute_mean_reward(self, x):
        return 1 / 2 * np.sin(13 * x) * np.sin(27 * x) + 0.5
