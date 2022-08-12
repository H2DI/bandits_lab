import numpy as np

# import cvxpy as cp
import math

from . import optim_utils


def draw_from_p(p, K):
    try:
        return np.random.choice(K, p=p)
    except ValueError:
        print(f"Tried to generate a sample from p = {p}, with sum {np.sum(p)}")
        raise ValueError


class AdvAlg:
    def __init__(self, K, **params):
        self.K = K
        self.alg_time = 0
        self.played_ps = []
        self.played_arms = []
        self.individual_rewards = []
        if "label" in params.keys():
            self.label = params["label"]

    def choose_p(self):
        pass

    def update(self, p, arm, reward):
        self.alg_time += 1

        self.played_ps.append(p)
        self.played_arms.append(arm)
        self.individual_rewards.append(reward)

    def play_once(self, bandit):
        p = self.choose_p()
        arm = draw_from_p(p, self.K)
        reward = bandit.play_arm(arm)
        self.update(p, arm, reward)

    def play_T_times(self, bandit, T):
        for _ in range(T):
            self.play_once(bandit)
        return

    def reset(self):
        self.alg_time = 0
        self.played_ps = []
        self.played_arms = []
        self.individual_rewards = []


class FTRLCanvas(AdvAlg):
    """
    This is a generic canvas for FTRL algorithms with any adaptive
    learning-rate scheme.

    For this to be complete, need to implement the functions choose_p,
    lr_update and estimate_method (although there is a default value).
    """

    def __init__(self, K, M=0, **params):
        super().__init__(K, **params)
        self.M = M  # parameter for the algorithm
        self.lr_value = np.inf
        self.finite_lr = False
        self.indiv_reward_estimates = []
        self.cum_reward_estimates = np.zeros(self.K)
        self.unif = np.ones(self.K) / self.K

    def choose_p(self):
        pass

    def estimate_method(self, p, arm, reward):
        """
            y_hat = M + (y_At - M ) / p_At
        """
        r = self.M * np.ones(self.K)
        r[arm] += (reward - self.M) / p[arm]
        return r

    def lr_update(self):
        pass

    def update(self, p, arm, reward):
        super().update(p, arm, reward)
        est = self.estimate_method(p, arm, reward)
        self.indiv_reward_estimates.append(est)
        self.cum_reward_estimates += est - self.M
        self.lr_update()

    def reset(self):
        super().reset()
        self.lr_value = np.inf
        self.finite_lr = False
        self.cum_reward_estimates = np.zeros(self.K)
        self.indiv_reward_estimates = []


class Exp3(FTRLCanvas):
    r"""
        The vanilla Exp3 algorithm with the usual learning rate scheme
        $\eta_t = log(K) / (K * \sqrt(t))$
    """

    def choose_p(self):
        if np.isinf(self.lr_value):
            p = self.unif
        else:
            logweights = self.lr_value * self.cum_reward_estimates
            max_logweight = np.max(logweights)
            temp = np.exp(logweights - max_logweight)
            p = temp / np.sum(temp)
        return p

    def lr_update(self):
        self.lr_value = np.sqrt(np.log(self.K) / (self.K * self.alg_time))


class AdaHedgeExp3(Exp3):
    """
        The AdaHedge learning rate update applied to the EXP3 algorithm.
    """

    def __init__(self, K, M=0, **params):
        super().__init__(K, M=M, **params)
        self.cum_mix_gap = 0
        self.D = np.log(self.K)
        self.mix_gaps = []

    def lr_update(self):
        p, reward_est = (
            self.played_ps[-1],
            self.indiv_reward_estimates[-1],
        )
        def_mix_gap = max(reward_est) - np.dot(p, reward_est)
        if np.isinf(self.lr_value):
            mix_gap = def_mix_gap
        else:
            # v_exp = self._vector_exp(self.lr_value * reward_est)
            v_exp = np.exp(self.lr_value * reward_est)
            if any(np.isinf(v_exp)):
                mix_gap = def_mix_gap
            else:
                mix_gap = -np.dot(p, reward_est) + 1 / self.lr_value * np.log(
                    np.dot(p, v_exp)
                )
        self.cum_mix_gap += mix_gap
        self.mix_gaps.append(mix_gap)

        if not self.finite_lr and np.isclose(self.cum_mix_gap, 0):
            self.lr_value = np.inf
        else:
            self.finite_lr = True
            self.lr_value = self.D / self.cum_mix_gap

    def reset(self):
        super().reset()
        self.cum_mix_gap = 0
        self.mix_gaps = []


class AdaHedgeExp3ExtraExp(AdaHedgeExp3):
    def __init__(self, K, **params):
        super().__init__(K, M=0, **params)
        self.beta = math.sqrt(10 * K * math.log(K))
        self.last_q = np.zeros(self.K)

    def choose_p(self):
        q = super().choose_p()
        self.last_q = q
        if self.alg_time < self.K:
            p = np.zeros(self.K)
            p[self.alg_time] = 1
        else:
            gamma_t = min(0.5, self.beta / math.sqrt(self.alg_time))
            p = (1 - gamma_t) * q + gamma_t * self.unif
        return p

    def lr_update(self):
        if self.alg_time < self.K:
            return
        p, reward_est = (
            self.last_q,
            self.indiv_reward_estimates[-1],
        )

        try:
            # v_exp = self._vector_exp(self.lr_value * reward_est)
            # v_exp = np.array([math.exp(x) for x in self.lr_value * reward_est])
            v_exp = np.exp(self.lr_value * reward_est)
            if np.any(np.isinf(v_exp)):
                mix_gap = np.max(reward_est) - np.dot(p, reward_est)
            else:
                mix_gap = -np.dot(p, reward_est) + 1 / self.lr_value * np.log(
                    np.dot(p, v_exp)
                )
        except OverflowError:
            mix_gap = np.max(reward_est) - np.dot(p, reward_est)

        # v_exp = self._vector_exp(self.lr_value * reward_est)
        # v_exp = np.array([math.exp(x) for x in self.lr_value * reward_est])
        # if np.any(np.isinf(v_exp)):
        #     mix_gap = np.max(reward_est) - np.dot(p, reward_est)
        # else:
        #     mix_gap = -np.dot(p, reward_est) + 1 / self.lr_value * np.log(
        #         np.dot(p, v_exp)
        #     )
        self.cum_mix_gap += mix_gap
        self.mix_gaps.append(mix_gap)
        if not self.finite_lr and np.isclose(self.cum_mix_gap, 0):
            self.lr_value = np.inf
        else:
            self.finite_lr = True
            self.lr_value = self.D / self.cum_mix_gap

    def update(self, p, arm, reward):
        super().update(p, arm, reward)
        if self.alg_time == self.K - 1:
            self.M = np.max(self.individual_rewards)
            self.cum_reward_estimates = np.zeros(self.K)
            self.indiv_reward_estimates = []

    def reset(self):
        super().reset()
        self.last_q = np.zeros(self.K)


class AdaFTRLTsallis(FTRLCanvas):
    """
        Requires that rewards be smaller than M.
    """

    def __init__(self, K, M=0, sym=False, proxy=False, **params):
        super().__init__(K, M=M, **params)
        self.sym = sym
        self.proxy = proxy
        self.cum_mix_gap = 0
        self.mix_gaps = []
        if self.sym:
            self.D = np.sqrt(self.K)
        else:
            self.D = 2 * np.sqrt(self.K) - 2

    def choose_p(self):
        if np.isinf(self.lr_value):
            return self.unif
        else:
            if self.sym:
                regularizer = optim_utils.Tsallis_1_2_sym(self.K)
            else:
                regularizer = optim_utils.Tsallis_1_2(self.K)
            return regularizer.reg_leader(-self.cum_reward_estimates, self.lr_value)

    def _mix_gap_comp(self, ell, p, eta):
        """
            Computes the generalized mixability gap
        """
        pvar, lvar, etavar = (
            cp.Parameter(self.K, nonneg=True),
            cp.Parameter(self.K),
            cp.Parameter(nonneg=True),
        )
        x = cp.Variable(self.K)

        tsallx, tsallp = -2 * cp.sum(cp.sqrt(x)), -2 * cp.sum(cp.sqrt(pvar))
        gradtsallp = -1 / cp.sqrt(pvar)
        breg = tsallx - tsallp - gradtsallp * (x - pvar)

        objective = lvar * (p - x) - 1 / etavar * breg

        pvar.value = p
        lvar.value = ell
        etavar.value = eta
        prob = cp.Problem(cp.Maximize(objective), [cp.sum(x) == 1, x >= 0])
        prob.solve()
        return x.value, objective.value

    def lr_update(self):
        p, arm, reward_est = (
            self.played_ps[-1],
            self.played_arms[-1],
            self.indiv_reward_estimates[-1],
        )
        pi = p[arm]
        ell = self.M - pi * reward_est[arm]  # true reward
        if np.isinf(self.lr_value):
            mix_gap = max(reward_est) - np.dot(p, reward_est)
        else:
            if self.sym:
                assert self.proxy
                mix_gap = min(
                    ell,
                    self.lr_value
                    * np.power(pi, -1 / 2)
                    * np.power(min(1, (1 - pi) / pi), 3 / 2)
                    * np.square(ell),
                )
            else:
                if self.proxy:
                    mix_gap = min(
                        ell, self.lr_value * np.power(pi, -1 / 2) * np.square(ell)
                    )
                else:
                    p_opt, mix_gap = self._mix_gap_comp(-reward_est, p, self.lr_value)
        self.mix_gaps.append(mix_gap)
        self.cum_mix_gap += mix_gap
        if np.isclose(self.cum_mix_gap, 0):
            self.lr_value = np.inf
        else:
            self.lr_value = self.D / self.cum_mix_gap

    def reset(self):
        super().reset()
        self.cum_mix_gap = 0
        self.mix_gaps = []


class ConvergenceError(Exception):
    pass


class FastAdaFTRLTsallis(AdaFTRLTsallis):
    """
    Implements Newton's method to compute the updates instead of using cvxpy
    """

    def __init__(self, K, M=0, proxy=True, speed=0.25, **params):
        super().__init__(K, M=M, proxy=proxy, **params)
        self.c = M
        # ideal theoretical value is speed=1, for stability of Newton's
        # method use .25
        self.speed = speed
        self.n_stops = 200

        self.verb = False

    def _comp_p(self, losses, speed, n_stops, verb=False):
        r""" Returns $ argmin_p <p, l >  + H_{{1/2}}(p)$ using Newton's method"""
        c = np.min(losses) - 1
        count = 0
        while count < n_stops:
            w = 1.0 / np.square(losses - c)
            c1 = c - speed * 2 * (np.sum(w) - 1) / np.sum(np.power(w, 3 / 2))
            if np.isclose(c, c1) and (np.isclose(np.sum(w), 1)):
                w = w / np.sum(w)
                return w
            else:
                c = c1
                count += 1
        raise ConvergenceError()

    def _fast_mix_gap_comp(self, ell, p, eta):
        """
            computes max( < p_t - p, l> - B_{H1/2}(p, p_t) )
            using _comp_p as a subroutine
        """
        h_of_p = -2 * np.sum(np.sqrt(p))
        grad_h_of_p = -1 / np.sqrt(p)
        p_opt = self._comp_p(-grad_h_of_p + eta * ell, self.speed, self.n_stops)
        value = np.dot(p, ell) + (1 / eta) * (
            h_of_p
            - np.dot(p, grad_h_of_p)
            + np.dot(p_opt, grad_h_of_p - eta * ell)
            - (-2 * np.sum(np.sqrt(p_opt)))
        )
        return p_opt, value

    def lr_update(self):
        p, arm, reward_est = (
            self.played_ps[-1],
            self.played_arms[-1],
            self.indiv_reward_estimates[-1],
        )
        pi = p[arm]
        ell = self.M - pi * reward_est[arm]  # true reward
        if np.isinf(self.lr_value):
            mix_gap = max(reward_est) - np.dot(p, reward_est)
        else:
            if self.sym:
                assert self.proxy
                mix_gap = min(
                    ell,
                    self.lr_value
                    * np.power(pi, -1 / 2)
                    * np.power(min(1, (1 - pi) / pi), 3 / 2)
                    * np.square(ell),
                )
            else:
                if self.proxy:
                    mix_gap = min(
                        ell, self.lr_value * np.power(pi, -1 / 2) * np.square(ell)
                    )
                else:
                    p_opt, mix_gap = self._fast_mix_gap_comp(
                        -reward_est, p, self.lr_value
                    )
        self.mix_gaps.append(mix_gap)
        self.cum_mix_gap += mix_gap
        if np.isclose(self.cum_mix_gap, 0):
            self.lr_value = np.inf
        else:
            self.lr_value = self.D / self.cum_mix_gap

    def choose_p(self):
        if np.isinf(self.lr_value):
            p = np.ones(self.K) / self.K
        else:
            p = self._comp_p(
                -self.lr_value * self.cum_reward_estimates,
                self.speed,
                self.n_stops,
                verb=self.verb,
            )
        return p


class FastFTRLTsallis(FastAdaFTRLTsallis):
    """
        Assumes rewards are smaller than M
    """

    def __init__(self, K, M=0, sym=False, **params):
        super().__init__(K, M=M, sym=sym, proxy=False, **params)

    def lr_update(self):
        self.lr_value = 4 * np.sqrt(1 / self.alg_time)


class FTRLTsallis(AdaFTRLTsallis):
    """
        Assumes rewards are smaller than M
    """

    def __init__(self, K, M=0, sym=False, **params):
        super().__init__(K, M=M, sym=sym, proxy=False, **params)

    def lr_update(self):
        self.lr_value = 2 * np.sqrt(1 / self.alg_time)


################################################################################


class AdaHedgeExp3Bounded(Exp3):
    """
    Assumes the rewards are smaller than M. Obsolete: no need to use the proxy
    for upper bounded rewards.
    """

    def __init__(self, K, M=0, **params):
        super().__init__(K, M=M, **params)
        self.cum_mix_gap = 0
        self.D = np.log(self.K)
        self.true_mix_gaps = []
        self.mix_gaps = []

    def lr_update(self):
        p, arm, reward = (
            self.played_ps[-1],
            self.played_arms[-1],
            self.individual_rewards[-1],
        )
        if np.isinf(self.lr_value):
            mix_gap = max(0, (self.M - reward))
            true_mix_gap = max((self.M - reward), (1 - self.K) * (self.M - reward))
        else:
            pi = p[arm]
            mix_gap = max(
                0, min(self.M - reward, self.lr_value * np.square(self.M - reward) / pi)
            )
            true_mix_gap = (self.M - reward) + (1 / self.lr_value) * np.log(
                (1 - pi) + pi * np.exp(-self.lr_value * (self.M - reward) / pi)
            )
        self.mix_gaps.append(mix_gap)
        self.true_mix_gaps.append(true_mix_gap)
        self.cum_mix_gap += mix_gap
        if np.isclose(self.cum_mix_gap, 0):
            self.lr_value = np.inf
        else:
            self.lr_value = self.D / self.cum_mix_gap

    def reset(self):
        super().reset()
        self.cum_mix_gap = 0
        self.true_mix_gaps = []
        self.mix_gaps = []
