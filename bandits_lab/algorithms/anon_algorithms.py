# import numpy as np
from ..bandit_definitions.div_p_bandits import AnonConstraints


class AnonAlg:
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

    def play_T_times(self, bandit, T):
        for _ in range(T):
            self.play_once(bandit)
        return

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


class AsDiv(AnonAlg):
    def __init__(self, K, M, cdistrib, alg, label="", **params):
        super().__init__(K, M, cdistrib, label=label)
        d = K * M
        self.setP = AnonConstraints(K, M, cdistrib)
        self.sub_div_alg = alg(d, self.setP, **params)

    def choose_arm(self):
        p = self.sub_div_alg.choose_p()
        return self.setP.a_from_p(p)

    def update(self, arm, reward, context):
        super().update(arm, reward, context)
        sub_arm = self.M * arm + context
        self.sub_div_alg.update(sub_arm, reward)

    def reset(self):
        super().reset()
        self.sub_div_alg.reset()
