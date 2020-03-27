import numpy as np
import cvxpy as cp

def in_simplex(p):
    delta = 0.00001
    return np.all(p >=-delta) and np.isclose(np.sum(p), 1)

def euclidean_simplex_proj(q):
    if in_simplex(q):
        return q
    temp = np.maximum(q, 0.0000001)
    return temp / np.sum(temp)

class Regularizer():
    """
        This is used to compute the minimizer in the FTRL method. The function f
        is a convex function from the simplex to the reals.
    """

    def __init__(self, d, cvx_function, **params):
        self.d = d
        self.cvx_function = cvx_function
        self.problem = None
        self._configure()


    def evaluate(self, p):
        return self.cvx_function(p).value

    def _configure(self):
        self.p = cp.Variable(self.d)
        self.L = cp.Parameter(self.d)
        self.lr = cp.Parameter()
        objective = self.cvx_function(self.p) + self.lr * cp.matmul(self.L, self.p)
        prob = cp.Problem(cp.Minimize(objective),
               [cp.sum(self.p) == 1,
                self.p >= 0])
        self.problem = prob

    def reg_leader(self, losses, lr):
        """
            Gradient descent over the simplex to minimize f(p) = dot(l, p) + f(p)
            Done in two steps : unconstrained optimization then projection
        """
        self.L.value = losses
        self.lr.value = lr
        self.problem.solve()
        return self.p.value

class Tsallis_1_2(Regularizer):
    def __init__(self, d, **params):
        def f(x):
            return -2*cp.sum(cp.sqrt(x))
        super().__init__(d, f, **params)

class Tsallis_1_2_sym(Regularizer):
    def __init__(self, d, **params):
        def f(x):
            return -cp.sum(cp.sqrt(x) + cp.sqrt(1-x))
        super().__init__(d, f, **params)


def mix_gap_comp(l, p, K, eta=1, i=0):
    """
    Conjecture : is smaller than etavar * l^2 / sqrt(pi) when l > 0
    """
    pvar, lvar, etavar = cp.Parameter(K, nonneg=True), cp.Parameter(), cp.Parameter(nonneg=True)
    x = cp.Variable(K)

    tsallx = -2*cp.sum(cp.sqrt(x))
    tsallp = -2*cp.sum(cp.sqrt(pvar))
    gradtsallp = - 1 / cp.sqrt(pvar)
    breg = tsallx - tsallp - gradtsallp * (x - pvar)

    obj =  lvar * ( 1  - x[i] / pvar[i]) - 1 / etavar * breg

    pvar.value = p
    lvar.value = l
    etavar.value = eta
    #print(obj.curvature)
    prob = cp.Problem(cp.Maximize(obj),
           [cp.sum(x) == 1,
            x >= 0])

    prob.solve()
    return x.value, obj.value
