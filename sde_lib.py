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

class VPSDE:
    def __init__(self, beta_min=0.1, beta_max=20.0, N=1000):
        """Variance Preserving SDE.
        
        Forward SDE: dx = -0.5 * beta(t) * x dt + sqrt(beta(t)) dw
        Has drift (shrinks the image) AND diffusion (adds noise).
        """
        
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.N = N

    def beta(self, t):
        """Noise rate at time t. Linear from beta_min to beta_max."""
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def marginal_prob(self, x, t):
        """Parameters of p(x_t | x_0).
        
        For VP SDE:
            mean = x_0 * exp(-0.5 * integral of beta)
            std  = sqrt(1 - exp(-integral of beta))
        
        The image shrinks (mean < x_0) while noise grows.
        """
        # Integral of beta(s) from 0 to t
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        mean = x * jnp.exp(log_mean_coeff)[:, None, None, None]
        std = jnp.sqrt(1.0 - jnp.exp(2.0 * log_mean_coeff))
        return mean, std[:, None, None, None]

    def prior_sampling(self, rng, shape):
        """Sample from p_T. For VP SDE, this is N(0, I)."""
        return jax.random.normal(rng, shape)


class subVPSDE:
    def __init__(self, beta_min=0.1, beta_max=20.0, N=1000):
        """Sub-Variance Preserving SDE.
        
        Same drift as VPSDE, but smaller diffusion coefficient.
        Better for likelihoods.
        """
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.N = N

    def beta(self, t):
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def marginal_prob(self, x, t):
        """Parameters of p(x_t | x_0).
        
        Same mean as VPSDE, but variance is (1 - exp(-integral))^2
        instead of (1 - exp(-integral)). Always smaller variance.
        """
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        mean = x * jnp.exp(log_mean_coeff)[:, None, None, None]
        std = 1.0 - jnp.exp(2.0 * log_mean_coeff)
        return mean, std[:, None, None, None]

    def prior_sampling(self, rng, shape):
        return jax.random.normal(rng, shape)