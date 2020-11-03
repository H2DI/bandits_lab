import numpy as np


class DAlg:
    def __init__(self, K, label=""):
        self.label = label
        self.K = K
        self.alg_time = 0

    def choose_arm(self):
        pass

    def update(self, arm, reward):
        self.alg_time += 1

    def play_once(self, bandit):
        arm = self.choose_arm()
        reward = bandit.play_arm(arm)
        self.update(arm, reward)
        return arm, reward

    def play_T_times(self, bandit, T):
        for _ in range(T):
            self.play_once(bandit)
        return

    def reset(self):
        self.alg_time = 0


class GenericIndexAlg(DAlg):
    def __init__(self, K, label=""):
        super().__init__(K, label=label)
        self.indices = np.ones(self.K)
        self.alg_n_plays = np.zeros(self.K)

        self.mean_rewards = [0 for _ in range(self.K)]

    def choose_arm(self):
        if self.alg_time < self.K:
            return self.alg_time
        return np.argmax(self.indices)

    def update(self, arm, reward):
        super().update(arm, reward)
        self.alg_n_plays[arm] += 1
        N = self.alg_n_plays[arm]
        self.mean_rewards[arm] = (self.mean_rewards[arm] * (N - 1) + reward) / N

    def reset(self):
        super().reset()
        self.indices = np.ones(self.K)
        self.alg_n_plays = np.zeros(self.K)
        self.mean_rewards = [0 for _ in range(self.K)]


class UCB_a(GenericIndexAlg):
    r"""
    UCB-anytime U_a(t) = \hat \mu_a(t) + sig * \sqrt{ 2 * \log (t) / N_a(t)}
    """

    def __init__(self, K, sig=1, label=""):
        super().__init__(K, label=label)
        self.sig = sig

    def update(self, arm, reward):
        super().update(arm, reward)
        if self.alg_time < self.K:
            return
        self.indices = self.mean_rewards + self.sig * np.sqrt(
            2 * np.log(self.alg_time) / (self.alg_n_plays)
        )


class MOSS_a(GenericIndexAlg):
    """ MOSS-anytime"""

    def __init__(self, K, sig=1, label=""):
        super().__init__(K, label=label)
        self.sig = sig

    def update(self, arm, reward):
        super().update(arm, reward)
        if self.alg_time < self.K:
            return
        u = np.maximum(np.ones(self.K, self.alg_time / (self.alg_n_plays * self.K)))
        self.indices = self.mean_rewards + self.sig * np.sqrt(
            2 * np.log(u) / (self.alg_n_plays)
        )


class MOSS_f(GenericIndexAlg):
    """ MOSS-horizon dependent"""

    def __init__(self, K, sig=1, label="", *, T):
        super().__init__(K, label=label)
        self.sig = sig
        self.T = T

    def update(self, arm, reward):
        super().update(arm, reward)
        if self.alg_time < self.K:
            return
        u = np.maximum(np.ones(self.K), self.T / (self.alg_n_plays * self.K))
        self.indices = self.mean_rewards + self.sig * np.sqrt(
            2 * np.log(u) / (self.alg_n_plays)
        )


class MaxUCB(UCB_a):
    def __init__(self, K, sig_init=0, label=""):
        super().__init__(K, sig=sig_init, label=label)
        self.sig_init = sig_init
        self.max_observed_reward = 0
        self.min_observed_reward = 0

    def update(self, arm, reward):
        super().update(arm, reward)
        self.max_observed_reward = max(self.max_observed_reward, reward)
        self.min_observed_reward = min(self.min_observed_reward, reward)
        self.sig = self.max_observed_reward - self.min_observed_reward

    def reset(self):
        super().reset()
        self.max_observed_reward = 0
        self.min_observed_reward = 0
        self.sig = self.sig_init


class EpsGreedy(DAlg):
    def __init__(self, K, label=None, epsilon=0):
        super().__init__(K, label=label)
        self.epsilon = epsilon
        self.alg_n_plays = np.zeros(self.K)
        self.mean_rewards = np.zeros(K)

    def choose_arm(self):
        eps = self.epsilon / (self.alg_time + 1)
        if np.random.rand() < eps:
            return np.random.randint(self.K)
        else:
            return np.argmax(self.mean_rewards)

    def update(self, arm, reward):
        super().update(arm, reward)
        self.alg_n_plays[arm] += 1
        N = self.alg_n_plays[arm]
        self.mean_rewards[arm] = (self.mean_rewards[arm] * (N - 1) + reward) / N

    def reset(self):
        super().reset()
        self.alg_n_plays = np.zeros(self.K)
        self.mean_rewards = np.zeros(self.K)


class RandomPlay(DAlg):
    def choose_arm(self):
        return np.random.randint(self.K)
