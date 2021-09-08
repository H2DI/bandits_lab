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
        u = np.maximum(np.ones(self.K), self.alg_time / (self.alg_n_plays * self.K))
        self.indices = self.mean_rewards + self.sig * np.sqrt(
            2 * np.log(u) / self.alg_n_plays
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


class GenericKlUCB(GenericIndexAlg):
    def __init__(self, K, label=""):
        super().__init__(K, label=label)

    def update(self, arm, reward):
        super().update(arm, reward)
        for i in range(self.K):
            n, t = self.alg_n_plays[i], self.alg_time
            expl = self.expl(t, n)
            self.indices[i] = ucb_kl(self.mean_rewards[i], expl / n)


class klUCB(GenericKlUCB):
    def expl(self, t, n):
        return np.log(t)


class klUCBplusplus(GenericKlUCB):
    def expl(self, t, n):
        return np.log(np.maximum(1, t / (self.K * n)))


class KLUCB(GenericIndexAlg):
    "Anytime version. Copied from PyMaBandits"

    def __init__(self, K, label=""):
        super().__init__(K, label=label)
        self.obs_dict = [{1: 0} for _ in range(self.K)]

    def expl(self, t, n):
        return np.log(t)

    def update(self, arm, reward):
        super().update(arm, reward)
        if reward in self.obs_dict[arm]:
            self.obs_dict[arm][reward] += 1
        else:
            self.obs_dict[arm][reward] = 1
        if self.alg_time <= self.K:
            return
        for i in range(self.K):
            n = self.alg_n_plays[i]
            expl = self.expl(self.alg_time, n) / n
            self.indices[i] = self.compute_KL_index(i, expl)

    def compute_KL_index(self, i, expl):
        if expl == 0:
            return self.mean_rewards[i]
        else:
            temp = np.array(list(self.obs_dict[i].values()))
            p = temp / np.sum(temp)
            V = np.array(list(self.obs_dict[i].keys()))
            q = self._maxEV(p, V, expl)
            return np.dot(q, V)

    def _maxEV(self, p, V, expl):
        Uq = np.zeros(np.size(p))
        Kb = p > 0
        K = p <= 0
        if np.any(K):
            eta_ = np.max(V[K])
            J = np.logical_and(K, (V == eta_))
            if eta_ > np.max(V[Kb]):
                y = np.dot(p[Kb], np.log(eta_ - V[Kb])) + np.log(
                    np.dot(p[Kb], 1 / (eta_ - V[Kb]))
                )
                if y < expl:
                    rb = np.exp(y - expl)
                    Uqtemp = p[Kb] / (eta_ - V[Kb])
                    Uq[Kb] = rb * Uqtemp / np.sum(Uqtemp)
                    Uq[J] = (1.0 - rb) / np.sum(J)
                    return Uq
        if np.any(np.abs(V[Kb] - V[Kb][0]) > 1e-8):
            eta_ = self._reseqp(p[Kb], V[Kb], expl)  # (eta = nu in the article)
            Uq = p / (eta_ - V)
            Uq = Uq / np.sum(Uq)
        else:
            Uq[Kb] = 1 / np.size(Kb)  # Case all values in V(Kb) are almost identical.
        return Uq

    def _reseqp(self, p, V, expl):
        mV = np.max(V)
        l = mV + 0.1
        tolerance = 1e-4

        if mV < np.min(V) + tolerance:
            return np.inf

        u = np.dot(p, (1.0 / (l - V)))
        y = np.dot(p, np.log(l - V)) + np.log(u) - expl

        while abs(y) > tolerance:
            yp = u - np.dot(p, np.square((1.0 / (l - V)))) / u  # derivative
            l = l - y / yp  # newton iteration
            if l < mV:
                l = (l + y / yp + mV) / 2  # unlikely, but not impossible

            u = np.dot(p, (1.0 / (l - V)))
            y = np.dot(p, np.log(l - V)) + np.log(u) - expl
        return l


class KLUCBPlusPlus(KLUCB):
    def expl(self, t, n):
        return np.log(np.maximum(1, t / (self.K * n)))


class KLUCBswitch(KLUCB):
    def switch(self, t):
        return np.power(t / self.K, 1 / 5)

    def expl(self, t, n):
        return np.log(np.maximum(1, t / (self.K * n)))

    def update(self, arm, reward):
        super(KLUCB, self).update(arm, reward)
        t = self.alg_time
        if reward in self.obs_dict[arm]:
            self.obs_dict[arm][reward] += 1
        else:
            self.obs_dict[arm][reward] = 1
        if t <= self.K:
            return
        for i in range(self.K):
            n = self.alg_n_plays[i]
            expl = self.expl(t, n) / n
            if n <= self.switch(t):
                self.indices[i] = super().compute_KL_index(i, expl)
            else:
                self.indices[i] = self.mean_rewards[i] + (1 / 2) * np.sqrt(2 * expl)


class KLUCBswitchProfile(KLUCBswitch):
    def __init__(self, K, label=""):
        super().__init__(K, label=label)
        self.profile = [[] for _ in range(K)]

    def switch(self, t):
        return np.power(t / self.K, 1 / 5)

    def expl(self, t, n):
        return np.log(np.maximum(1, t / (self.K * n)))

    def update(self, arm, reward):
        super(KLUCB, self).update(arm, reward)
        t = self.alg_time
        if reward in self.obs_dict[arm]:
            self.obs_dict[arm][reward] += 1
        else:
            self.obs_dict[arm][reward] = 1
        if t <= self.K:
            return
        for i in range(self.K):
            n = self.alg_n_plays[i]
            expl = self.expl(t, n) / n
            if n <= self.switch(t):
                self.profile[i].append(0)
                self.indices[i] = super().compute_KL_index(i, expl)
            else:
                self.profile[i].append(1)
                self.indices[i] = self.mean_rewards[i] + (1 / 2) * np.sqrt(2 * expl)


class KLUCBswitchPro(KLUCB):
    def switch(self, t):
        return np.power(t / self.K, 1 / 5)

    def update(self, arm, reward):
        super(KLUCB, self).update(arm, reward)
        t = self.alg_time
        if reward in self.obs_dict[arm]:
            self.obs_dict[arm][reward] += 1
        else:
            self.obs_dict[arm][reward] = 1
        if t <= self.K:
            return
        for i in range(self.K):
            n = self.alg_n_plays[i]
            expl = self.expl(t, n) / n
            if n <= self.switch(t):
                self.indices[i] = super().compute_KL_index(i, expl)
            else:
                self.indices[i] = ucb_kl(self.mean_rewards[i], expl)


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


# class IMED(DAlg):


def kl(p, q):
    assert (0 <= p <= 1) and (0 <= q <= 1)
    if q == 0 or q == 1:
        return np.inf
    elif p == 0 or p == 1:
        return p * np.log(1 / q) + (1 - p) * np.log(1 / (1 - q))
    else:
        return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def ucb_kl(mu, exploration, precision=0.001):
    assert 0 <= mu <= 1
    if mu > 1 - precision:
        return 1.0
    r = 1.0
    delta = 1.0 - mu
    counter = 2 * np.log(1 / precision)
    while counter > 0:
        counter -= 1
        delta = delta / 2.0
        if kl(mu, r) > exploration + precision:
            r = r - delta
        elif kl(mu, r) < exploration - precision:
            r = r + delta
        else:
            return r
    return r


class ConvergenceError(Exception):
    pass
