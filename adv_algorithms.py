import numpy as np
import bandit_definitions as band_defs
import optim_utils as opt_ut
import cvxpy as cp

class AdvAlg():
    def __init__(self, K, **params):
        if 'label' in params:
            self.label = params['label']
        else:
            self.label = "No label specified"
        self.K = K

        self.alg_time = 0
        self.played_ps = []
        self.played_arms = []
        self.individual_rewards = []

    def choose_p(self):
        pass

    def update(self, p, arm, reward):
        self.alg_time += 1

        self.played_ps.append(p)
        self.played_arms.append(arm)
        self.individual_rewards.append(reward)

    def play_once(self, bandit):
        p = self.choose_p()
        arm =  band_defs.draw_from_p(p, self.K)
        reward = bandit.play_arm(arm)
        self.update(p, arm, reward)

    def reset(self):
        self.alg_time = 0
        self.played_ps = []
        self.played_arms = []
        self.individual_rewards = []

class FTRLCanvas(AdvAlg):
    """
    This is a generic canvas for FTRL/OMD algorithms with any adaptive
    learning-rate scheme. (Not sure this is appropriate for OMD as currently
    this only stores the cumulative rewards of every arms.)

    For this to be complete, need to implement the functions choose_p,
    lr_update and estimate_method (although there is a default value).

    Need to implement extra-exploration
    """
    def __init__(self, K, M=0, **params):
        super().__init__(K, **params)
        self.M = M # parameter for the algorithm
        self.lr_value = np.inf
        self.indiv_reward_estimates = []
        self.cum_reward_estimates = np.zeros(self.K)

    def choose_p(self):
        pass

    def estimate_method(self, p, arm, reward):
        """
         y_hat = M + (y_At - M ) / p_At
        """
        assert(p[arm] > 0)
        r = self.M * np.ones(self.K)
        r[arm] += (reward - self.M)/ p[arm]
        return r

    def lr_udpate():
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
        self.cum_reward_estimates = np.zeros(self.K)
        self.indiv_reward_estimates = []

class Exp3(FTRLCanvas):
    """
    The vanilla Exp3 algorithm with the usual learning rate scheme.
    """

    def choose_p(self):
        if np.isinf(self.lr_value):
            p = np.ones(self.K) / self.K
        else:
            logweights = self.lr_value * self.cum_reward_estimates
            temp = np.exp(logweights - np.max(logweights))
            p = temp / np.sum(temp)
        return p


    def lr_update(self):
        self.lr_value = np.sqrt(np.log(self.K) / (self.K*self.alg_time))

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
        p, arm, reward_est = self.played_ps[-1], self.played_arms[-1], self.indiv_reward_estimates[-1]
        if np.isinf(self.lr_value):
            mix_gap = max(reward_est) - np.dot(p, reward_est)
        else:
            mix_gap = -np.dot(p, reward_est) + 1 / self.lr_value * np.log(np.dot(p, np.exp(self.lr_value * reward_est)))
        self.cum_mix_gap += mix_gap
        self.mix_gaps.append(mix_gap)
        if np.isclose(self.cum_mix_gap, 0):
            self.lr_value = np.inf
        else:
            self.lr_value = self.D / self.cum_mix_gap

    def reset(self):
        super().reset()
        self.cum_mix_gap = 0
        self.mix_gaps = []

class FTRL_w_reg(FTRLCanvas):
    """
        FTRL with a regularizer
    """
    def __init__(self, K, regularizer, M=0, **params):
        super().__init__(K, M=M, **params)
        self.regularizer = regularizer

    def choose_p(self):
        if np.isinf(self.lr_value):
            return np.ones(self.K)/self.K
        else:
            return self.regularizer.reg_leader(-self.cum_reward_estimates, self.lr_value)

    def lr_update(self):
        self.lr_value = np.sqrt(1 / self.alg_time)

class AdaFTRLTsallis(FTRL_w_reg):
    """
        Requires that rewards be smaller than M.
    """
    def __init__(self, K, M=0, sym=False, proxy=False, **params):# Needs rewards smaller than M
        if sym:
            super().__init__(K, opt_ut.Tsallis_1_2_sym(K), M=M, **params)
        else:
            super().__init__(K, opt_ut.Tsallis_1_2(K), M=M, **params)
        self.sym = sym
        self.proxy = proxy
        self.cum_mix_gap = 0
        self.mix_gaps = []
        if self.sym:
            self.D = np.sqrt(self.K)
        else:
            self.D = 2*np.sqrt(self.K)

    def _mix_gap_comp(self, l , p, eta):
        """
            Computes the generalized mixability gap
        """
        pvar, lvar, etavar = cp.Parameter(self.K, nonneg=True), cp.Parameter(self.K), cp.Parameter(nonneg=True)
        x = cp.Variable(self.K)

        tsallx = -2*cp.sum(cp.sqrt(x))
        tsallp = -2*cp.sum(cp.sqrt(pvar))
        gradtsallp = - 1 / cp.sqrt(pvar)
        breg = tsallx - tsallp - gradtsallp * (x - pvar)

        objective =  lvar * (p -x) - 1 / etavar * breg

        pvar.value = p
        lvar.value = l
        etavar.value = eta
        prob = cp.Problem(cp.Maximize(objective),
               [cp.sum(x) == 1,
                x >= 0])

        prob.solve()
        return x.value, objective.value

    def lr_update(self):
        p, arm, reward_est = self.played_ps[-1], self.played_arms[-1], self.indiv_reward_estimates[-1]
        pi = p[arm]
        l = self.M - pi*reward_est[arm]
        if np.isinf(self.lr_value):
                mix_gap = max(reward_est) - np.dot(p, reward_est)
        else:
            if self.sym:
                assert(self.proxy)
                mix_gap = min(l, self.lr_value * np.power(pi, -1/2) * np.power(min(1, (1 - pi)/pi), 3/2) * np.square(l))
            else:
                if self.proxy:
                    mix_gap = min(l, self.lr_value * np.power(pi, -1/2) * np.square(l))
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

#############################################
#############################################
#############################################
#############################################
#############################################

class AdaHedgeExp3Bounded(Exp3):#Obsolete :
    """
    Assumes the rewards are smaller than M. Obsolete: no need to use the proxy for upper bounded rewards.
    """
    def __init__(self, K, M=0, **params):
        super().__init__(K, M=M, **params)
        self.cum_mix_gap = 0
        self.D = np.log(self.K)
        self.true_mix_gaps = []
        self.mix_gaps = []

    def lr_update(self):
        p, arm, reward = self.played_ps[-1], self.played_arms[-1], self.individual_rewards[-1]
        if np.isinf(self.lr_value):
            mix_gap = max(0, (self.M - reward))
            true_mix_gap = max((self.M - reward), (1- self.K)*(self.M - reward))
        else:
            pi = p[arm]
            mix_gap = max(0, min(self.M - reward, self.lr_value * np.square(self.M - reward) / pi))
            true_mix_gap = (self.M - reward) + (1 / self.lr_value) * np.log( (1 - pi) + pi*np.exp(-self.lr_value * (self.M - reward) / pi))
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
