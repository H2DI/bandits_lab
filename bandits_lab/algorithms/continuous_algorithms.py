import numpy as np
from . import gen_algorithms as gen_alg

#################


class ContAlg:
    def __init__(self, label=""):
        self.label = label
        self.alg_time = 0
        pass

    def choose_arm(self):
        pass

    def update(self, arm, reward):
        self.alg_time += 1
        pass

    def play_once(self, bandit):
        arm = self.choose_arm()
        reward = bandit.play_arm(arm)
        self.update(arm, reward)

    def play_T_times(self, bandit, T):
        for _ in range(T):
            self.play_once(bandit)
        return

    def reset(self):
        self.alg_time = 0


class Discretizer:
    def __init__(self, K):
        self.K = K

    def discrete_to_continuous(self, i):
        pass


class UniformDiscretizer(Discretizer):
    def discrete_to_continuous(self, i):
        return np.random.uniform(i / self.K, (i + 1) / self.K)


class AbstractCAB(ContAlg):
    """
        Turn a discrete bandit algorithm into a continuous bandit algorithm.
         discretizer
        A canonical example is the uniform discretization
    """

    def __init__(self, *, K, discrete_alg, discretizer, label=""):
        super().__init__(label=label)
        self.K = K
        self.discrete_alg = discrete_alg
        self.discretizer = discretizer
        self.last_discrete_arm_played = None

    def choose_arm(self):
        discrete_arm = self.discrete_alg.choose_arm()
        self.last_discrete_arm_played = discrete_arm
        return self.discretizer.discrete_to_continuous(discrete_arm)

    def update(self, contarm, reward):
        super().update(contarm, reward)
        self.discrete_alg.update(self.last_discrete_arm_played, reward)

    def reset(self):
        super().reset()
        self.discrete_alg.reset()


class CAB1(AbstractCAB):
    """ Standard CAB1 algorithm with MOSS indices """

    def __init__(self, moss_sig=1, label="", *, K, T):
        discrete_alg = gen_alg.MOSS_f(K, T=T, sig=moss_sig)
        super().__init__(
            K=K,
            discrete_alg=discrete_alg,
            discretizer=UniformDiscretizer(K),
            label=label,
        )


#################
class MeDZO(ContAlg):
    def __init__(self, B, label=""):
        super().__init__(label=label)
        self.p = int(np.log2(B)) + 1
        self.reset()

    def _initialize(self):
        K = self.current_K
        discrete_moss_alg = gen_alg.MOSS_f(
            K=self.current_K, sig=0.25, T=self.current_DeltaT
        )
        self.current_alg = AbstractCAB(
            K=K, discrete_alg=discrete_moss_alg, discretizer=UniformDiscretizer(K)
        )

    def choose_arm(self):
        return self.current_alg.choose_arm()

    def update(self, arm, reward):
        self.current_alg.update(arm, reward)
        self.alg_time += 1
        self.played_arm_in_current_period += [arm]
        self.time_in_period += 1
        if self.time_in_period >= self.current_DeltaT:
            self.up_next_period()
        return

    def up_next_period(self):
        self.all_played_arms_per_period.append(
            np.array(self.played_arm_in_current_period)
        )
        self.played_arm_in_current_period = []
        self.time_in_period = 1

        self.current_period += 1
        self.current_K = int(self.current_K / 2)
        self.current_DeltaT *= 2
        K = self.current_K

        class LocDisc(Discretizer):
            def discrete_to_continuous(self, i):
                if i < self.K:
                    return np.random.uniform(i / self.K, (i + 1) / self.K)
                else:
                    return np.random.choice(self.all_played_arms_per_period[i - self.K])

        discrete_moss_alg = gen_alg.MOSS_f(
            K=self.current_K, sig=0.25, T=self.current_DeltaT
        )
        self.current_alg = AbstractCAB(
            K=K, discrete_alg=discrete_moss_alg, discretizer=LocDisc(K)
        )

    def reset(self):
        self.alg_time = 0
        self.time_in_period = 1
        self.current_DeltaT = np.power(2, self.p + 2)
        self.current_K = int(np.power(2, self.p))
        self.current_period = 0
        self.played_arm_in_current_period = []
        self.all_played_arms_per_period = []

        self._initialize()


#################
class SubRoutine(ContAlg):
    """ From Locatelli and Carpentier 2018 """

    def __init__(self, exponent, delta, label=""):
        super().__init__(label=label)
        self.exponent = exponent
        self.delta = delta
        self.tlalpha = 1

        self.alg_time = 0
        self.current_depth = 0
        self.active_centers = []
        self.active_centers_means = []

        self.time_in_current_phase = 0
        self.remaining_centers = []

        self._initialize()

    def _initialize(self):
        self.delta_l = self.delta
        self.blalpha = 1

        self.active_centers = [0.5]
        self.remaining_centers = self.active_centers
        # In order to freeze the values when we remove remaining centers
        self.active_centers = np.array(self.active_centers)
        self.active_centers_means = [0]

    def choose_arm(self):
        if self.remaining_centers[0]:
            return self.remaining_centers[0]
        else:
            print("No arm was selected in SubRoutine")

    def update(self, arm, reward):
        super().update(arm, reward)
        self.time_in_current_phase += 1
        self.active_centers_means[-1] += reward
        assert self.time_in_current_phase < self.tlalpha + 1
        if (
            self.time_in_current_phase >= self.tlalpha
        ):  # we have drawn the current active cell enough times
            self.active_centers_means[-1] /= self.tlalpha
            self.time_in_current_phase = 0
            self.remaining_centers.pop(0)
            if self.remaining_centers:
                self.active_centers_means += [0]
            else:
                self.update_next_depth()

    def update_next_depth(self):
        self.current_depth += 1
        h = np.power(1 / 2, (self.current_depth + 1))
        self.delta_l = self.delta * (h ** 2)
        self.blalpha = np.power(h, self.exponent)
        self.tlalpha = 0.5 * np.log(1 / self.delta_l) / (self.blalpha ** 2)

        B = 2 * (np.sqrt(np.log(1 / self.delta_l) / (2 * self.tlalpha)) + self.blalpha)
        # Compute the active centers at next depth
        M = max(self.active_centers_means)
        new_centers = []
        assert len(self.active_centers_means) == len(self.active_centers)
        for i, x in enumerate(self.active_centers):
            if (M - self.active_centers_means[i]) <= B:
                new_centers += [x - h, x + h]
        self.remaining_centers = new_centers
        self.active_centers = np.array(self.remaining_centers)  # to freeze
        self.active_centers_means = [0]

    def reset(self):
        super().reset()
        self.tlalpha = 1

        self.current_depth = 0
        self.active_centers = []
        self.active_centers_means = []

        self.time_in_current_phase = 0
        self.remaining_centers = []
        self._initialize()


#################
class TreeHOO:
    """ Tree object for the HOO algorithm """

    def __init__(self, point=0.5, depth=1):
        self.point = point
        self.depth = depth
        self.children = (None, None)
        self.is_leaf = True

        self.n_visits = 0
        self.mean = None
        self.U = None
        self.B = float("inf")

    def expand(self):
        self.is_leaf = False
        n = self.depth
        pleft, pright = (
            self.point - np.power(0.5, n + 1),
            self.point + np.power(0.5, n + 1),
        )
        self.children = TreeHOO(pleft, n + 1), TreeHOO(pright, n + 1)

    def find_max_b(self, path):
        """ Returns the full path visited to find the most promising arm """
        path.append(self)
        if self.is_leaf:
            return path
        else:
            l, r = self.children
            if l.B > r.B:
                return l.find_max_b(path)
            else:
                return r.find_max_b(path)
        return


class HOO(ContAlg):
    """ HOO algorithm """

    def __init__(self, T, alpha, nu=1, label=""):
        super().__init__(label=label)
        self.T = T
        self.nu = nu
        self.alpha = alpha
        self.reset()

    def reset(self):
        self.alg_time = 1
        self.last_path = None
        self.tree = TreeHOO()

    def choose_arm(self):
        path = self.tree.find_max_b([])
        self.last_path = path
        return path[-1].point

    def update(self, arm, reward):
        for node in self.last_path:
            if node.is_leaf:
                node.expand()
            n = node.n_visits
            node.n_visits = n + 1
            if node.mean:
                node.mean = n / (n + 1) * node.mean + reward / (n + 1)
            else:
                node.mean = reward
            node.U = (
                node.mean
                + np.sqrt(np.log(self.T) / (2 * node.n_visits))
                + self.nu * np.power(0.5, (node.depth) * self.alpha)
            )
            node.B = min(node.U, max(node.children[0].B, node.children[1].B))


###############


class RandomPlayer(ContAlg):
    def choose_arm(self):
        return np.random.random()


class AbstractCAB2(ContAlg):
    """
        Discretize the action space and play according to the MOSS algorithm
    """

    # distributions_function is a function (0,..., K-1) -> [0, 1]
    # delta is the moss confidence parameter
    def __init__(self, distributions_function, K, delta):
        super().__init__()
        self.K = K
        self.distributions_function = distributions_function
        self.delta = delta

        self.alg_time = 0
        self.indices = np.ones(self.K)
        self.alg_n_plays = np.zeros(self.K)
        self.second_max_arm = 0
        self.next_play = 0  # next_play is the discrete arm

        self.mean_rewards = [0 for _ in range(self.K)]

    def choose_arm(self):
        sampled_arm = self.distributions_function(self.next_play)
        return sampled_arm

    def update(self, contarm, reward):
        arm = self.next_play

        self.alg_time += 1
        self.alg_n_plays[arm] += 1

        N = self.alg_n_plays[arm]
        self.mean_rewards[arm] = (self.mean_rewards[arm] * (N - 1) + reward) / N
        self.indices[arm] = self.mean_rewards[arm] + np.sqrt(
            max(0, np.log(1.0 / ((self.delta ** 2) * N))) / (2 * N)
        )
        if (self.alg_time - 1) < self.K:
            self.next_play = self.alg_time - 1
        elif (self.alg_time - 1) == self.K:
            best_arms = np.argpartition(self.indices, -2)[-2:]
            if self.indices[best_arms[0]] > self.indices[best_arms[1]]:
                self.second_max_arm = best_arms[1]
                self.next_play = best_arms[0]
            else:
                self.second_max_arm = best_arms[0]
                self.next_play = best_arms[1]
        elif self.indices[arm] < self.indices[self.second_max_arm]:
            self.next_play = self.second_max_arm
            best_arms = np.argpartition(self.indices, -2)[-2:]
            if self.indices[best_arms[0]] > self.indices[best_arms[1]]:
                assert self.indices[best_arms[0]] == self.indices[self.second_max_arm]
                self.second_max_arm = best_arms[1]
            else:
                assert self.indices[best_arms[1]] == self.indices[self.second_max_arm]
                self.second_max_arm = best_arms[0]

    def reset(self):
        self.alg_time = 0
        self.indices = np.ones(self.K)
        self.alg_n_plays = np.zeros(self.K)
        self.second_max_arm = 0
        self.next_play = 0  # next_play is the discrete arm

        self.mean_rewards = [0 for _ in range(self.K)]
        return


# Next may be interesting to get better performance on MOSS: smart updates of
# the indices
# class DiscretizationMOSS2:  # delta >= 1/K corresponds to FTL
#     def __init__(self, h, delta):
#         self.h = h
#         self.delta = delta
#         self.K = int(1.0 / h) + 1
#
#         self.alg_time = 0
#         self.indices = np.ones(self.K)
#         self.alg_n_plays = np.zeros(self.K)
#         self.second_max_arm = 0
#         self.next_play = 0
#
#         self.mean_rewards = [
#             0 for _ in range(self.K)
#         ]  # delta is the moss confidence parameter. take (K/T)
#
#     def __discr2cont(self, i):
#         return np.random.uniform(i * self.h, min(1.0, (i + 1) * self.h))
#
#     def choose_arm(self):
#         return self.__discr2cont(self.next_play)
#
#     def update(self, contarm, reward):
#         arm = self.next_play
#
#         self.alg_time += 1
#         self.alg_n_plays[arm] += 1
#
#         N = self.alg_n_plays[arm]
#         self.mean_rewards[arm] = (self.mean_rewards[arm] * (N - 1) + reward) / N
#         self.indices[arm] = self.mean_rewards[arm] + np.sqrt(
#             max(0, np.log(1.0 / ((self.delta ** 2) * N))) / (2 * N)
#         )
#         if (self.alg_time - 1) < self.K:
#             self.next_play = self.alg_time - 1
#         elif (self.alg_time - 1) == self.K:
#             best_arms = np.argpartition(self.indices, -2)[-2:]
#             if self.indices[best_arms[0]] > self.indices[best_arms[1]]:
#                 self.second_max_arm = best_arms[1]
#                 self.next_play = best_arms[0]
#             else:
#                 self.second_max_arm = best_arms[0]
#                 self.next_play = best_arms[1]
#         elif self.indices[arm] < self.indices[self.second_max_arm]:
#             self.next_play = self.second_max_arm
#             best_arms = np.argpartition(self.indices, -2)[-2:]
#             if self.indices[best_arms[0]] > self.indices[best_arms[1]]:
#                 assert best_arms[0] == self.second_max_arm
#                 self.second_max_arm = best_arms[1]
#             else:
#                 assert best_arms[1] == self.second_max_arm
#                 self.second_max_arm = best_arms[0]
#
#     def play_once(self, bandit):
#         arm = self.choose_arm()
#         reward = bandit.play_arm(arm)
#         self.update(arm, reward)
#
#     def reset(self):
#         self.alg_time = 0
#         self.indices = np.ones(self.K)
#         self.alg_n_plays = np.zeros(self.K)
#         return
