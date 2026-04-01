import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from utils import batch_mul

def get_sde(config):
    N = config.training.sde_N
    sampler_steps = config.sampler.sampler_steps if 'sampler' in config else None
    if sampler_steps == 2000:
        N = 2000
    if config.training.sde == 'vesde':
        return VESDE(config.model.sigma_min, config.model.sigma_max, N), 1e-5
    elif config.training.sde == 'vpsde':
        factor = 2. if sampler_steps == 2000 else 1.
        return VPSDE(config.model.beta_min/factor, config.model.beta_max/factor, N), 1e-3
    else:
        return subVPSDE(config.model.beta_min, config.model.beta_max, N), 1e-3

class VPSDE:
    def __init__(self, beta_min=0.1, beta_max=20.0, N=1000):
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = N
        self.T = 1.0
        self.discrete_betas = jnp.linspace(beta_min / N, beta_max / N, N)
        self.alphas = 1.0 - self.discrete_betas
        self.alphas_cumprod = jnp.cumprod(self.alphas)

    def marginal_prob(self, x, t):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean = batch_mul(jnp.exp(log_mean_coeff), x)
        std = jnp.sqrt(1.0 - jnp.exp(2.0 * log_mean_coeff))
        return mean, std

    def sde(self, x, t):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * batch_mul(beta_t, x)
        diffusion = jnp.sqrt(beta_t)
        return drift, diffusion

    def prior_sampling(self, rng, shape):
        return jax.random.normal(rng, shape)

    def prior_logp(self, x):
        logprob = jsp.stats.norm.logpdf(x)
        axes = tuple(range(1, x.ndim))
        return jnp.sum(logprob, axis=axes)

    def t_to_idx(self, t):
        return (t * (self.N - 1) / self.T).astype(jnp.int32)

    def discretize(self, x, t):
        timestep = self.t_to_idx(t)
        beta = self.discrete_betas[timestep]
        alpha = self.alphas[timestep]
        f = batch_mul(jnp.sqrt(alpha), x) - x
        G = jnp.sqrt(beta)
        return f, G

    def reverse_sde(self, x, t, score, probability_flow=False):
        f, g = self.sde(x, t)
        score_factor = 0.5 if probability_flow else 1
        rev_f = f - batch_mul(g ** 2, score * score_factor)
        return rev_f, jnp.zeros_like(g) if probability_flow else g


class subVPSDE:
    def __init__(self, beta_min=0.1, beta_max=20.0, N=1000):
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = N
        self.T = 1.0

    def marginal_prob(self, x, t):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean = batch_mul(jnp.exp(log_mean_coeff), x)
        std = 1.0 - jnp.exp(2.0 * log_mean_coeff)
        return mean, std

    def sde(self, x, t):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * batch_mul(beta_t, x)
        discount = 1.0 - jnp.exp(-2 * self.beta_0 * t - (self.beta_1 - self.beta_0) * t ** 2)
        diffusion = jnp.sqrt(beta_t * discount)
        return drift, diffusion

    def prior_sampling(self, rng, shape):
        return jax.random.normal(rng, shape)

    def prior_logp(self, x):
        logprob = jsp.stats.norm.logpdf(x)
        axes = tuple(range(1, x.ndim))
        return jnp.sum(logprob, axis=axes)

    def t_to_idx(self, t):
        return (t * (self.N - 1) / self.T).astype(jnp.int32)

    def discretize(self, x, t):
        dt = 1.0 / self.N
        drift, diffusion = self.sde(x, t)
        return drift * dt, diffusion * jnp.sqrt(dt)

    def reverse_sde(self, x, t, score, probability_flow=False):
        f, g = self.sde(x, t)
        score_factor = 0.5 if probability_flow else 1
        rev_f = f - batch_mul(g ** 2, score * score_factor)
        return rev_f, jnp.zeros_like(g) if probability_flow else g


class VESDE:
    def __init__(self, sigma_min=0.01, sigma_max=50.0, N=1000):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.N = N
        self.T = 1.0
        self.discrete_sigmas = jnp.exp(
            np.linspace(np.log(sigma_min), np.log(sigma_max), N)
        )

    def marginal_prob(self, x, t):
        std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        return x, std

    def sde(self, x, t):
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        drift = jnp.zeros_like(x)
        diffusion = sigma * jnp.sqrt(2 * (jnp.log(self.sigma_max) - jnp.log(self.sigma_min)))
        return drift, diffusion

    def prior_sampling(self, rng, shape):
        return jax.random.normal(rng, shape) * self.sigma_max

    def prior_logp(self, x):
        logprob = jsp.stats.norm.logpdf(x, scale=self.sigma_max)
        axes = tuple(range(1, x.ndim))
        return jnp.sum(logprob, axis=axes)

    def t_to_idx(self, t):
        return (t * (self.N - 1) / self.T).astype(jnp.int32)

    def discretize(self, x, t):
        timestep = self.t_to_idx(t)
        sigma = self.discrete_sigmas[timestep]
        adj = jnp.where(timestep == 0, jnp.zeros_like(sigma), self.discrete_sigmas[timestep - 1])
        return jnp.zeros_like(x), jnp.sqrt(sigma ** 2 - adj ** 2)

    def reverse_sde(self, x, t, score, probability_flow=False):
        f, g = self.sde(x, t)
        score_factor = 0.5 if probability_flow else 1
        rev_f = f - batch_mul(g ** 2, score * score_factor)
        return rev_f, jnp.zeros_like(g) if probability_flow else g
