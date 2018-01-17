import math
try:
    import numpypy
except ImportError:
    pass

# Distributions without priors


class Uniform(object):
    def __init__(self, K):
        self.K = K
        self.count = 0

    def increment(self, k, initialize=False):
        self.count += 1

    def decrement(self, k):
        self.count -= 1

    def prob(self, k):
        if k >= self.K: return 0
        return 1./self.K

    def log_likelihood(self, full=False):
        return - self.count * math.log(self.K)

    def resample_hyperparemeters(self, n_iter):
        return (0, 0)

    def __repr__(self):
        return 'Uniform(K={self.K}, count={self.count})'.format(self=self)