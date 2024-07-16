import numpy as np


class DivPAlg:
    def __init__(self, K, setP, label=None):
        self.K = K
        self.setP = setP
        self.label = label
        self.alg_time = 0

    def choose_p(self):
        pass

    def play_once(self, divp_bandit):
        p = self.choose_p()
        arm, reward = divp_bandit.play_p(p)
        self.update(arm, reward)

    def play_T_times(self, bandit, T):
        for _ in range(T):
            self.play_once(bandit)
        return

    def update(self, arm, reward):
        self.alg_time += 1

    def reset(self):
        self.alg_time = 0


# kl  Utilities
def kl(p, q):
    assert (0 <= p <= 1) and (0 <= q <= 1)
    if q == 0 or q == 1:
        return 10000.0
    elif p == 0 or p == 1:
        return p * np.log(1 / q) + (1 - p) * np.log(1 / (1 - q))
    else:
        return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def ucb_kl(mu, n, t, precision=0.0001):
    assert 0 <= mu <= 1
    if mu > 1 - precision:
        return 1
    r = 1.0
    delta = 1 - mu
    while True:
        delta = delta / 2
        if kl(mu, r) > np.log(t) / n + precision:
            r = r - delta
        elif kl(mu, r) < np.log(t) / n - precision:
            r = r + delta
        else:
            return r


class DivIndexPolicies(DivPAlg):
    """Generic index policy for diversity-preserving bandits"""

    def __init__(self, K, setP, label=""):
        super().__init__(K, setP, label=label)

        self.alg_n_plays = np.zeros(self.K)
        self.mean_rewards = np.zeros(self.K)
        self.indices = np.ones(self.K)

    def choose_p(self):
        return self.setP.argmax_dot(self.indices).x

    def update(self, arm, reward):
        super().update(arm, reward)
        self.alg_n_plays[arm] += 1
        N = self.alg_n_plays[arm]
        self.mean_rewards[arm] = (self.mean_rewards[arm] * (N - 1) + reward) / N

    def reset(self):
        super().reset()
        self.alg_n_plays = np.zeros(self.K)
        self.mean_rewards = np.zeros(self.K)
        self.indices = np.ones(self.K)


class DivPUCB(DivIndexPolicies):
    r"""Index $U_a(t) = \hat \mu_a(t) + sqrt( 2 \ln t  / N_a(t))$"""

    def __init__(self, K, setP, sig=1 / 2, label="", **params):
        super().__init__(K, setP, label=label)
        self.sig = sig

    def update(self, arm, reward):
        super().update(arm, reward)
        self.indices = np.minimum(
            1,
            self.mean_rewards
            + self.sig
            * np.sqrt(8 * np.log(self.alg_time) / np.maximum(self.alg_n_plays, 1)),
        )


class DivPklUCB(DivIndexPolicies):
    r"""
    Index $U_a(t)$ is such that
    $kl(\hat \mu_a(t) , U_a(t)) \approx \log t / N_a(t)$
    """

    def __init__(self, K, setP, label="", **params):
        super().__init__(K, setP, label=label)

    def update(self, arm, reward):
        super().update(arm, reward)
        self.indices = np.array(
            [
                ucb_kl(self.mean_rewards[a], max(self.alg_n_plays[a], 1), self.alg_time)
                for a in range(self.K)
            ]
        )


class DivPEpsGreedy(DivPAlg):
    def __init__(self, K, setP, label="", **params):
        super().__init__(K, setP, label=label)
        self.epsilon = params["epsilon"]
        self.alg_n_plays = np.zeros(self.K)
        self.mean_rewards = np.zeros(K)

    def choose_p(self):
        eps = self.epsilon / (self.alg_time + 1)
        if np.random.rand() < eps:
            return self.setP.feasible
        else:
            return self.setP.argmax_dot(self.mean_rewards).x

    def update(self, arm, reward):
        super().update(arm, reward)
        self.alg_n_plays[arm] += 1
        N = self.alg_n_plays[arm]
        self.mean_rewards[arm] = (self.mean_rewards[arm] * (N - 1) + reward) / N

    def reset(self):
        super().reset()
        self.alg_n_plays = np.zeros(self.K)
        self.mean_rewards = np.zeros(self.K)


class L1OFUL(DivPAlg):
    """
    Treats the diversity-preserving bandit problem as a linear bandit problem and plays
    the L1-OFUL algorithm.
    """

    def __init__(self, K, setP, label="", **params):
        super().__init__(K, setP, label=label)
        self.alg_n_plays = np.zeros(K)

        self.delta = params["delta"]
        self.bt = np.zeros((K, 1))
        self.muhat = np.zeros((K, 1))
        self.played_ps = []
        self.vt = np.identity(K)
        self.det_vt = 1
        self.vt_inv = np.identity(K)

        # used for the confidence interval (i.e. at the corners of the simplex)
        self.sq_pts = np.ones(K)

    def choose_p(self):
        if self.alg_time == 0:
            p = self.setP.feasible
            self.played_ps.append(p)
            return p

        if self.delta == 0:
            sqrt_beta = 1 + np.sqrt(2 * np.log(self.alg_time * self.det_vt))
        else:
            sqrt_beta = 1 + np.sqrt(2 * np.log(self.det_vt / self.delta))
        current_max = 0
        current_p = np.ones(self.K) / self.K
        for i in range(self.K):
            ei = np.array([1 * (j == i) for j in range(self.K)]) / (self.sq_pts[i])
            loc_muhat = self.muhat.reshape(self.K)

            optimizer = self.setP.argmax_dot(
                loc_muhat + ei * np.sqrt(self.K) * sqrt_beta
            )
            if optimizer.fun > current_max:
                current_max = optimizer.fun
                current_p = optimizer.x

            optimizer = self.setP.argmax_dot(
                loc_muhat - ei * np.sqrt(self.K) * sqrt_beta
            )
            if optimizer.fun > current_max:
                current_max = optimizer.fun
                current_p = optimizer.x
        p = current_p
        self.played_ps.append(p)
        return p

    def update(self, arm, reward):
        super().update(arm, reward)
        self.alg_n_plays[arm] += 1

        p = self.played_ps[-1]
        self.sq_pts += np.array([pa * pa for pa in p])
        p = p.reshape(self.K, 1)

        r1update = np.matmul(p, p.T)
        self.vt += r1update
        temp = np.matmul(self.vt_inv, p)
        temp2 = 1 + np.dot(p.T, temp)[0, 0]
        inverse_update = np.matmul(temp, np.matmul(p.T, self.vt_inv)) / temp2
        self.det_vt *= temp2
        self.vt_inv = self.vt_inv - inverse_update

        self.bt += reward * p
        self.muhat = np.matmul(self.vt_inv, self.bt)

    def reset(self):
        super().reset()
        K = self.K
        self.alg_n_plays = np.zeros(K)

        self.bt = np.zeros((K, 1))
        self.muhat = np.zeros((K, 1))
        self.played_ps = []
        self.vt = np.identity(K)
        self.det_vt = 1
        self.vt_inv = np.identity(K)

        self.sq_pts = np.ones(K)


class ConstantSampling(DivPAlg):
    def __init__(self, K, setP, label="", **params):
        super().__init__(K, setP, label=label)
        if "point" in params:
            self.point = params["point"]
        else:
            self.point = self.setP.feasible

    def choose_p(self):
        return self.point
