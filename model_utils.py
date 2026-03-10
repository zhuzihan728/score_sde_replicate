import jax.numpy as jnp


def get_score_fn(sde, model, params, train=False, continuous=True):
    """
    The raw model output and the score have different relationships
    depending on the SDE type. This wrapper handles that conversion.
    """

    def score_fn(x, t):
        # Raw model output
        output = model.apply(params, x, t, train=train)

        if continuous:
            # Get std at time t
            _, std = sde.marginal_prob(jnp.zeros_like(x), t)
            score = -output / std
        else:
            score = output

        return score

    return score_fn