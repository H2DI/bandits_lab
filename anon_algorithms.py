import numpy as np
from bandit_definitions import *

class AnonAlg():
    def __init__(self, K, M, cdistrib, label=""):
        self.K = K
        self.M = M
        self.cdistrib = cdistrib
        self.label = label

        self.alg_time = 0

    def choose_arm(self):
        pass

    def play_once(self, anon_bandit):
        arm = self.choose_arm()
        reward, context = anon_bandit.play_arm(arm)
        self.update(arm, reward, context)

    def update(self, arm, reward, context):
        self.alg_time += 1
        pass

    def reset(self):
        self.alg_time = 0
        pass

class Ignore(AnonAlg):
    def __init__(self, K, M, cdistrib, alg, label="", **params):
        super().__init__(K, M, cdistrib, label=label)
        self.sub_alg = alg(K, **params)

    def choose_arm(self):
        return self.sub_alg.choose_arm()

    def update(self, arm, reward, context):
        super().update(arm, reward, context)
        self.sub_alg.update(arm, reward)

    def reset(self):
        super().reset()
        self.sub_alg.reset()

class AsFair(AnonAlg):
    def __init__(self, K, M, cdistrib, alg, label="", **params):
        super().__init__(K, M, cdistrib, label=label)
        d = K*M
        self.C = AnonConstraints(K, M, cdistrib)
        self.sub_fair_alg = alg(d, self.C, **params)

    def choose_arm(self):
        p = self.sub_fair_alg.choose_p()
        return self.C.a_from_p(p)

    def update(self, arm, reward, context):
        super().update(arm, reward, context)
        sub_arm = self.M*arm + context
        self.sub_fair_alg.update(sub_arm, reward)

    def reset(self):
        super().reset()
        self.sub_fair_alg.reset()
