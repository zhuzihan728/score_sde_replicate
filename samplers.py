import flax
import jax
import jax.numpy as jnp
from scipy import integrate
import numpy as np

from sde import VPSDE, subVPSDE, VESDE
from utils import batch_mul

class Predictor:
    def __init__(self, sde=None, score_fn=None, probability_flow=False):
        self.sde = sde
        self.score_fn = score_fn
        self.probability_flow = probability_flow

    def update_fn(self, rng, x, t):
        return x, x

class Corrector:
    def __init__(self, sde=None, score_fn=None, snr=None, n_steps=None):
        self.sde = sde
        self.score_fn = score_fn
        self.snr = snr
        self.n_steps = n_steps

    def update_fn(self, rng, x, t):
        return x, x

class EulerMaruyamaPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def update_fn(self, rng, x, t):
        dt = 1. / self.sde.N
        z = jax.random.normal(rng, x.shape)
        f, G = self.sde.reverse_sde(x, t, self.score_fn(x,t), self.probability_flow)
        x_mean = x - f*dt
        x = x_mean + batch_mul(G, jnp.sqrt(dt)*z)
        return x, x_mean
    
class ReverseDiffusionPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def update_fn(self, rng, x, t):
        f, G = self.sde.reverse_sde(x, t, self.score_fn(x,t), self.probability_flow)
        z = jax.random.normal(rng, x.shape)
        x_mean = x - f
        x = x_mean + batch_mul(G, z)
        return x, x_mean
    
class AncestralSamplingPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def vpsde_update(self, rng, x, t):
        timestep = (t * (self.sde.N - 1) / self.sde.T).astype(jnp.int32)
        beta = self.sde.discrete_betas[timestep]
        z = jax.random.normal(rng, x.shape)
        x_mean = batch_mul(x+batch_mul(beta, self.score_fn(x,t)), 1./jnp.sqrt(1-beta))
        x = x_mean + batch_mul(jnp.sqrt(beta), z)
        return x, x_mean
    
    def vesde_update(self, rng, x, t):
        timestep = (t * (self.sde.N - 1) / self.sde.T).astype(jnp.int32)
        sigma = self.sde.discrete_sigmas[timestep]
        sigma_prev = self.sde.discrete_sigmas[timestep-1] if timestep > 0 else jnp.zeros_like(t)
        z = jax.random.normal(rng, x.shape)
        x_mean = x-batch_mul(sigma**2-sigma_prev**2, self.score_fn(x,t))
        x = x_mean + batch_mul(jnp.sqrt((sigma**2-sigma_prev**2)/sigma**2), z)
        return x, x_mean
    
    def update_fn(self, rng, x, t):
        if isinstance(self.sde, VPSDE) or isinstance(self.sde, subVPSDE):
            return self.vpsde_update(rng, x, t)
        return self.vesde_update(rng, x, t)
    
# Note: Algorithms 4 and 5 in the paper actually correspond to vanilla Langevin dynamcs    
class LangevinCorrector(Corrector):
    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__(sde, score_fn, snr, n_steps)

    def update_fn(self, rng, x, t):
        if isinstance(self.sde, VPSDE) or isinstance(self.sde, subVPSDE):
            timestep = (t * (self.sde.N - 1) / self.sde.T).astype(jnp.int32)
            alpha = self.sde.alphas[timestep]
        else:
            alpha = jnp.ones_like(t)

        def iteration(step, val):
            rng, x, x_mean = val
            score = self.score_fn(x, t)
            rng, step_rng = jax.random.split(rng)
            z = jax.random.normal(step_rng, x.shape)
            score_norm = jnp.linalg.norm(score.reshape((score.shape[0], -1)), axis=-1).mean()
            score_norm = jax.lax.pmean(score_norm, axis_name='batch')
            noise_norm = jnp.linalg.norm(z.reshape((z.shape[0],-1)), axis=-1).mean()
            noise_norm = jax.lax.pmean(noise_norm, axis_name='batch')
            step_size = (self.snr*noise_norm/score_norm)**2*alpha
            x_mean = x + batch_mul(step_size, score)
            x = x_mean + batch_mul(jnp.sqrt(2*step_size), z)
            return rng, x, x_mean

        _, x, x_mean = jax.lax.fori_loop(0, self.n_steps, iteration, (rng, x, x))
        return x, x_mean
    
class AnnealedLangevinCorrector(Corrector):
    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__(sde, score_fn, snr, n_steps)

    def update_fn(self, rng, x, t):
        if isinstance(self.sde, VPSDE) or isinstance(self.sde, subVPSDE):
            timestep = (t * (self.sde.N - 1) / self.sde.T).astype(jnp.int32)
            alpha = self.sde.alphas[timestep]
        else:
            alpha = jnp.ones_like(t)

        _, std = self.sde.marginal_prob(x, t)

        def iteration(step, val):
            rng, x, x_mean = val
            score = self.score_fn(x, t)
            rng, step_rng = jax.random.split(rng)
            z = jax.random.normal(step_rng, x.shape)
            step_size = (self.snr*std)**2*alpha
            x_mean = x + batch_mul(step_size, score)
            x = x_mean + batch_mul(jnp.sqrt(2*step_size), z)
            return rng, x, x_mean

        _, x, x_mean = jax.lax.fori_loop(0, self.n_steps, iteration, (rng, x, x))
        return x, x_mean
    
def pc_sampler(
        sde, shape, predictor, corrector, inverse_scaler, n_steps=1_000, denoise=True, epsilon=1e-3
    ):
    # For predictor only sampling, set corrector = Corrector()
    # For corrector only sampling, set predictor = Predictor()

    def sampler_batch(rng):
        rng, step_rng = jax.random.split(rng)
        x = sde.prior_sampling(step_rng, shape)
        timesteps = jnp.linspace(sde.T, epsilon, sde.N)

        # Predictor - Corrector order is opposite to algorithm 1 in the paper
        # This is because jax.lax.fori_loop only works forward
        def iteration(step, val):
            rng, x, x_mean = val
            t = timesteps[step]
            vec_t = jnp.full((shape[0],), t)
            rng, step_rng = jax.random.split(rng)
            x, x_mean = corrector.update_fn(step_rng, x, vec_t)
            rng, step_rng = jax.random.split(rng)
            x, x_mean = predictor.update_fn(step_rng, x, vec_t)
            return rng, x, x_mean
        
        _, x, x_mean = jax.lax.fori_loop(0, sde.N, iteration, (rng, x, x))
        return (inverse_scaler(x_mean) if denoise else x), sde.N*(n_steps+1)

    return jax.pmap(sampler_batch, axis_name='batch')

def ode_sampler(
        sde, score_fn, shape, inverse_scaler, denoise=False, rtol=1e-5, atol=1e-5, method='RK45', eps=1e-3
    ):

    @jax.pmap
    def denoise_update_fn(rng, x):
        predictor = ReverseDiffusionPredictor(sde, score_fn, probability_flow=False)
        vec_e = jnp.full((shape[0],), eps)
        _, x = predictor.update_fn(rng, x, vec_e)
        return x
    
    @jax.pmap
    def drift_fn(x, t):
        f, _ = sde.reverse_sde(x, t, score_fn(x,t), probability_flow=True)
        return f

    def sampler(prng, z=None):
        rng = flax.jax_utils.unreplicate(prng)
        rng, step_rng = jax.random.split(rng)
        x = z if z else sde.prior_sampling(step_rng, (jax.local_device_count(),)+shape)

        def ode(t,x):
            vec_t = jnp.full((x.shape[0],x.shape[1]), t)
            return drift_fn(x, vec_t)

        solution = integrate.solve_ivp(ode, (sde.T, eps), x, rtol=rtol, atol=atol, method=method)
        nfe = solution.nfev
        x = jnp.asarray(solution.y[:, -1]).reshape((jax.local_device_count(),) + shape)

        if denoise:
            rng, *step_rng = jax.random.split(rng, jax.local_device_count() + 1)
            step_rng = jnp.asarray(step_rng)
            x = denoise_update_fn(step_rng, x)

        x = inverse_scaler(x)
        return x, nfe

    return ode_sampler