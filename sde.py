"""VE / VP / sub-VP SDEs for score-based generative modelling."""
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from utils import batch_mul

import library.sde_lib as sde_lib

def get_sde(config):
    N = config.training.sde_N
    if config.sampler.sampler_steps == 2000:
        N = 2000
    if config.training.sde == 'vesde':
        return VESDE(config.model.sigma_min, config.model.sigma_max, N), 1e-5
    elif config.training.sde == 'vpsde':
        factor = 2. if config.sampler.sampler_steps == 2000 else 1.
        return VPSDE(config.model.beta_min/factor, config.model.beta_max/factor, N), 1e-3
    else:
        return subVPSDE(config.model.beta_min, config.model.beta_max, N), 1e-3
    
def get_old_sde(config):
    N = config.training.sde_N
    if config.training.sde == 'vesde':
        return sde_lib.VESDE(config.model.sigma_min, config.model.sigma_max, N)
    elif config.training.sde == 'vpsde':
        return sde_lib.VPSDE(config.model.beta_min, config.model.beta_max, N)
    else:
        return sde_lib.subVPSDE(config.model.beta_min, config.model.beta_max, N)

class VPSDE:
    """β(t) = β_min + t(β_max − β_min), variance-preserving SDE."""

    def __init__(self, beta_min=0.1, beta_max=20.0, N=1000):
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = N
        self.T = 1.0
        self.discrete_betas = jnp.linspace(beta_min / N, beta_max / N, N)
        if float(self.discrete_betas.max()) >= 1.0:
            raise ValueError(
                f"VPSDE: discrete_betas exceed 1 (max={float(self.discrete_betas.max()):.4f}). "
                f"Reduce beta_max or increase N (need beta_max/N < 1, got {beta_max}/{N})."
            )
        self.alphas = 1.0 - self.discrete_betas
        self.alphas_cumprod = jnp.cumprod(self.alphas)

    # ── Forward SDE ─────────────────────────────────────────────────────────

    def marginal_prob(self, x, t):
        """p_t(x_t | x_0): mean = exp(∫β)·x_0, std = sqrt(1 − exp(2∫β))."""
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean = batch_mul(jnp.exp(log_mean_coeff), x)
        std = jnp.sqrt(1.0 - jnp.exp(2.0 * log_mean_coeff))
        return mean, std

    def sde(self, x, t):
        """Drift f and diffusion g of dx = f dt + g dW."""
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * batch_mul(beta_t, x)
        diffusion = jnp.sqrt(beta_t)
        return drift, diffusion

    # ── Reverse / prior ─────────────────────────────────────────────────────

    def prior_sampling(self, rng, shape):
        """Sample from p_T ≈ N(0, I)."""
        return jax.random.normal(rng, shape)

    def prior_logp(self, x):
        logprob = jsp.stats.norm.logpdf(x)
        axes = tuple(range(1,x.ndim))
        return jnp.sum(logprob, axis=axes)

    def t_to_idx(self, t):
        return (t * (self.N - 1) / self.T).astype(jnp.int32)

    def discretize(self, x, t):
        """DDPM discretization used by the reverse-diffusion predictor."""
        timestep = self.t_to_idx(t)
        beta = self.discrete_betas[timestep]
        alpha = self.alphas[timestep]
        f = batch_mul(jnp.sqrt(alpha), x) - x
        G = jnp.sqrt(beta)
        return f, G

    def reverse_sde(self, x, t, score, probability_flow=False):
        """Reverse-time SDE drift and diffusion given the score."""
        f, g = self.sde(x, t)
        score_factor = 0.5 if probability_flow else 1
        rev_f = f - batch_mul(g ** 2, score*score_factor)
        return rev_f, g


class subVPSDE:
    """Sub-VP SDE: same drift as VP but reduced diffusion (good at likelihoods)."""

    def __init__(self, beta_min=0.1, beta_max=20.0, N=1000):
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = N
        self.T = 1.0

    # ── Forward SDE ─────────────────────────────────────────────────────────

    def marginal_prob(self, x, t):
        """p_t(x_t | x_0): std = 1 − exp(2∫β) (not square-rooted, per paper)."""
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean = batch_mul(jnp.exp(log_mean_coeff), x)
        std = 1.0 - jnp.exp(2.0 * log_mean_coeff)
        return mean, std

    def sde(self, x, t):
        """Drift f and diffusion g of dx = f dt + g dW."""
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * batch_mul(beta_t, x)
        discount = 1.0 - jnp.exp(-2 * self.beta_0 * t - (self.beta_1 - self.beta_0) * t ** 2)
        diffusion = jnp.sqrt(beta_t * discount)
        return drift, diffusion

    # ── Reverse / prior ─────────────────────────────────────────────────────

    def prior_sampling(self, rng, shape):
        """Sample from p_T ≈ N(0, I)."""
        return jax.random.normal(rng, shape)
    
    def prior_logp(self, x):
        logprob = jsp.stats.norm.logpdf(x)
        axes = tuple(range(1,x.ndim))
        return jnp.sum(logprob, axis=axes)

    def t_to_idx(self, t):
        return (t * (self.N - 1) / self.T).astype(jnp.int32)

    def discretize(self, x, t):
        dt = 1.0 / self.N
        drift,diffusion = self.sde(x,t)
        f = drift*dt
        G = diffusion*jnp.sqrt(dt)
        return f, G

    def reverse_sde(self, x, t, score, probability_flow=False):
        """Reverse-time SDE drift and diffusion given the score."""
        f, g = self.sde(x, t)
        score_factor = 0.5 if probability_flow else 1
        rev_f = f - batch_mul(g ** 2, score*score_factor)
        return rev_f, g


class VESDE:
    """σ(t) = σ_min * (σ_max / σ_min)^t  ·  data is never scaled (mean = x)."""

    def __init__(self, sigma_min=0.01, sigma_max=50.0, N=1000):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.N = N
        self.T = 1.0
        self.discrete_sigmas = jnp.exp(
            np.linspace(np.log(sigma_min), np.log(sigma_max), N)
        )

    # ── Forward SDE ─────────────────────────────────────────────────────────

    def marginal_prob(self, x, t):
        """p_t(x_t | x_0): mean = x_0, std = σ(t)."""
        std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        return x, std

    def sde(self, x, t):
        """Drift f and diffusion g of the forward SDE dx = f dt + g dW."""
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        drift = jnp.zeros_like(x)
        diffusion = sigma * jnp.sqrt(2 * (jnp.log(self.sigma_max) - jnp.log(self.sigma_min)))
        return drift, diffusion

    # ── Reverse / prior ─────────────────────────────────────────────────────

    def prior_sampling(self, rng, shape):
        """Sample from p_T ≈ N(0, σ_max² I)."""
        return jax.random.normal(rng, shape) * self.sigma_max

    def prior_logp(self, x):
        logprob = jsp.stats.norm.logpdf(x, scale=self.sigma_max)
        axes = tuple(range(1,x.ndim))
        return jnp.sum(logprob, axis=axes)

    def t_to_idx(self, t):
        return (t * (self.N - 1) / self.T).astype(jnp.int32)

    def discretize(self, x, t):
        """SMLD discretization used by the reverse-diffusion predictor."""
        timestep = self.t_to_idx(t)
        sigma = self.discrete_sigmas[timestep]
        adj = jnp.where(timestep == 0, jnp.zeros_like(sigma), self.discrete_sigmas[timestep - 1])
        return jnp.zeros_like(x), jnp.sqrt(sigma ** 2 - adj ** 2)

    def reverse_sde(self, x, t, score, probability_flow=False):
        """Reverse-time SDE drift and diffusion given the score."""
        f, g = self.sde(x, t)
        score_factor = 0.5 if probability_flow else 1
        rev_f = f - batch_mul(g ** 2, score*score_factor)
        return rev_f, g
