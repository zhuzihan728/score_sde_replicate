import jax
import jax.numpy as jnp


def get_loss_fn(sde, model, train, reduce_mean=False,  continuous=False):

    def loss_fn(rng, params, batch):
        """Compute one training loss.
        
        Args:
            rng: JAX random key
            params: Model parameters
            batch: Clean images, shape [B, H, W, C], already scaled to [-1,1]
        
        Returns:
            loss: Scalar loss value
        """
        # Step 1: Sample random times
        rng, step_rng = jax.random.split(rng)
        if continuous:
            t = jax.random.uniform(step_rng, (batch.shape[0],),
                                   minval=1e-5, maxval=1.0)
        else:
            t = jax.random.randint(step_rng, (batch.shape[0],),
                                   minval=0, maxval=sde.N)
            t = t.astype(jnp.float32)  # Flax layers expect float input
        # Step 2: Sample noise
        rng, step_rng = jax.random.split(rng)
        z = jax.random.normal(step_rng, batch.shape)

        # Step 3: Perturb the images
        mean, std = sde.marginal_prob(batch, t)
        perturbed = mean + std * z

        # Step 4: Model predicts the score
        score = model.apply(params, perturbed, t, train=train)

        # Step 5: Compute the loss
        # If model is perfect: score = -z / std
        # So: score * std + z = 0
        losses = jnp.square(score * std + z)

        # Reduce over pixels (H, W, C)
        losses = losses.reshape(losses.shape[0], -1)  # [B, H*W*C]
        if reduce_mean:
            losses = jnp.mean(losses, axis=-1)  # Average over pixels
        else:
            losses = jnp.sum(losses, axis=-1)   # Sum over pixels

        # Average over batch
        loss = jnp.mean(losses)
        return loss

    return loss_fn