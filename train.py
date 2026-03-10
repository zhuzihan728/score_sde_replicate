import jax
import jax.numpy as jnp
import optax
import functools
from config import get_config_local
from datasets import get_dataset, get_data_scaler, get_data_inverse_scaler
from sde_lib import VESDE
from model import UNet
from losses import get_loss_fn
import matplotlib.pyplot as plt


def train(config):
    # SETUP 
    rng = jax.random.PRNGKey(42)

    # Data
    train_ds, _ = get_dataset(config)
    scaler = get_data_scaler(config.data.centered)
    inverse_scaler = get_data_inverse_scaler(config.data.centered)
    train_iter = iter(train_ds)

    # SDE
    sde = VESDE(
        sigma_min=config.model.sigma_min,
        sigma_max=config.model.sigma_max,
        N=config.model.num_scales
    )

    # Model
    model = UNet(config=config)
    rng, init_rng = jax.random.split(rng)
    dummy_x = jnp.ones((config.training.batch_size, 
                         config.data.image_size, 
                         config.data.image_size, 
                         config.data.num_channels))
    dummy_t = jnp.ones(config.training.batch_size)
    params = model.init(init_rng, dummy_x, dummy_t)

    # EMA params: a copy that gets slowly updated
    ema_params = params

    # Optimizer: Adam with warmup
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.training.learning_rate,
        warmup_steps=config.training.warmup,
        decay_steps=config.training.n_iters
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.training.grad_clip),
        optax.adam(learning_rate=schedule)
    )
    opt_state = optimizer.init(params)

    # Loss function
    loss_fn = get_loss_fn(sde, model, train=True, 
                          reduce_mean=config.training.reduce_mean)

    # TRAIN STEP 
    @jax.jit
    def train_step(rng, params, opt_state, batch):
        """One gradient update."""
        # Compute loss and gradients
        loss, grads = jax.value_and_grad(
            lambda p: loss_fn(rng, p, batch)
        )(params)

        # Update parameters
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        return loss, new_params, new_opt_state

    # TRAINING LOOP 
    losses = []
    print(f"Training for {config.training.n_iters} iterations...")

    for step in range(1, config.training.n_iters + 1):
        # Get batch and scale to [-1, 1]
        batch = jnp.array(scaler(next(train_iter).numpy()))

        # Train step
        rng, step_rng = jax.random.split(rng)
        loss, params, opt_state = train_step(step_rng, params, opt_state, batch)

        # EMA update
        ema_params = jax.tree.map(
            lambda ema, new: config.model.ema_rate * ema + (1 - config.model.ema_rate) * new,
            ema_params, params
        )

        losses.append(float(loss))

        if step % 10 == 0:
            print(f"Step {step}/{config.training.n_iters}, Loss: {loss:.2f}")

    # PLOT LOSS CURVE
    plt.figure(figsize=(10, 4))
    plt.plot(losses)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig('loss_curve.png', dpi=150)
    plt.show()
    print("Saved loss_curve.png")

    return ema_params, model, sde


if __name__ == '__main__':
    config = get_config_local()
    ema_params, model, sde = train(config)