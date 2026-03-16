import jax
import jax.numpy as jnp
from score import get_score_fn
from utils import batch_mul


def get_loss_fn(sde, model, train, reduce_mean=False, continuous=True):

    def loss_fn(rng, params, batch):
        rng, step_rng = jax.random.split(rng)
        if continuous:
            t = jax.random.uniform(step_rng, (batch.shape[0],), minval=1e-5, maxval=1.0)
        else:
            # sample t from {1/N, 2/N, ..., 1}, n time steps.
            t = jax.random.randint(step_rng, (batch.shape[0],),
                                   minval=1, maxval=sde.N + 1).astype(jnp.float32) / sde.N

        rng, step_rng = jax.random.split(rng)
        z = jax.random.normal(step_rng, batch.shape)

        # not exact for discrete time steps
        mean, std = sde.marginal_prob(batch, t)
        perturbed = mean + batch_mul(std, z)

        score = get_score_fn(sde, model, params, train=train, continuous=continuous)(perturbed, t)

        # score = -z / std
        losses = jnp.square(batch_mul(std, score) + z)
        losses = losses.reshape(losses.shape[0], -1)
        losses = jnp.mean(losses, axis=-1) if reduce_mean else jnp.sum(losses, axis=-1)
        return jnp.mean(losses)

    return loss_fn
