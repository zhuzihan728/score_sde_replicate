import jax.numpy as jnp
from sde import VESDE, VPSDE
from utils import batch_mul


def get_score_fn(sde, model, params, train=False, continuous=True, rng=None):

    def score_fn(x, t):
        rngs = {'dropout': rng} if (train and rng is not None) else {}

        if isinstance(sde, VESDE):
            sigma = sde.marginal_prob(x, t)[1]
            if continuous:
                time_cond = jnp.log(sigma)          # log(sigma) for continuous
            else:
                time_cond = sde.t_to_idx(t).astype(jnp.float32)  # integer index for discrete DDPM
            output = model.apply(params, x, time_cond, train=train, rngs=rngs)
            return batch_mul(1.0 / sigma, output)

        if continuous or not isinstance(sde, VPSDE):
            time_cond = t * 999
            _, std = sde.marginal_prob(jnp.zeros_like(x), t)
        else:
            time_cond = t * sde.N
            timestep = sde.t_to_idx(t)
            std = jnp.sqrt(1.0 - sde.alphas_cumprod[timestep])

        output = model.apply(params, x, time_cond, train=train, rngs=rngs)
        return -batch_mul(1.0 / std, output)

    return score_fn
