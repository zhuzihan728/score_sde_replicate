import jax.numpy as jnp
import jax
import numpy as np


class VESDE:
    def __init__(self, sigma_min=0.01, sigma_max=50.0, N=1000):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.N = N

    def sigma(self, t):
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    def marginal_prob(self, x, t):
        std = self.sigma(t)
        mean = x
        return mean, std[:, None, None, None]

    def prior_sampling(self, rng, shape):
        return jax.random.normal(rng, shape) * self.sigma_max
