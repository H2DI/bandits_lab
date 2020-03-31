import numpy as np
from bandit_definitions import *
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
        arm =  draw_from_p(p, self.K)
        reward = bandit.play_arm(arm)
        self.update(p, arm, reward)

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
        try:
            assert(p[arm] > 0)
        except ValueError:
            print(p, arm, p[arm], np.sum(p), np.isclose(p, np.sum(p)))
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
    The vanilla Exp3 algorithm with the usual learning rate scheme : log(K) / (K *sqrt(T))
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
