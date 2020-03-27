import numpy as np

class DAlg():
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

    def reset(self):
        self.alg_time = 0

class UCB(DAlg):
    def __init__(self, K, sig=1, label=""):
        super().__init__(K, label=label)
        self.sig = sig
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
        self.mean_rewards[arm] = (self.mean_rewards[arm]*(N -1) + reward)/ N
        self.indices = self.mean_rewards + self.sig * np.sqrt(2*np.log(self.alg_time)) /((self.alg_n_plays + 0.001))

    def reset(self):
        super().reset()
        self.indices = np.ones(self.K)
        self.alg_n_plays = np.zeros(self.K)
        self.mean_rewards = [0 for _ in range(self.K)]

class EpsGreedy(DAlg):
    def __init__(self, K, label="", **params):
        super().__init__(K, label=label)
        self.epsilon = params['epsilon']
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
        self.mean_rewards[arm] = (self.mean_rewards[arm]*(N -1) + reward) / N

    def reset(self):
        super().reset()
        self.alg_n_plays = np.zeros(self.K)
        self.mean_rewards = np.zeros(self.K)

class RandomPlay(DAlg):
    def choose_arm(self):
        return np.random.randint(self.K)
