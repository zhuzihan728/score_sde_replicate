import jax.numpy as jnp
from sde import VESDE, VPSDE
from utils import batch_mul


def get_score_fn(sde, model, params, train=False, continuous=True):

    def score_fn(x, t):
        if isinstance(sde, VESDE):
            sigma = sde.marginal_prob(x, t)[1]       # [B]
            output = model.apply(params, x, jnp.log(sigma), train=train)
            return batch_mul(1.0 / sigma, output)  # predicted score = output (normalized) / sigma

        if continuous or not isinstance(sde, VPSDE):
            time_cond = t * 999
            _, std = sde.marginal_prob(jnp.zeros_like(x), t)
        else:
            time_cond = t * sde.N # tembed encode {1, 2, ..., N}
            timestep = sde.t_to_idx(t) # [B] int in [0, N-1]
            std = jnp.sqrt(1.0 - sde.alphas_cumprod[timestep])

        output = model.apply(params, x, time_cond, train=train)
        return -batch_mul(1.0 / std, output) # predicted score = -output (noise) / std

    return score_fn
