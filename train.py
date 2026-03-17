import argparse, functools, pathlib, time
import numpy as np
import jax, jax.numpy as jnp
import optax
import tensorflow as tf
import orbax.checkpoint as ocp

from config import get_config
from datasets import get_dataset, get_data_scaler, get_data_inverse_scaler
from sde import get_sde, VESDE
from model import UNet
from losses import get_loss_fn
from score import get_score_fn
from samplers import EulerMaruyamaPredictor, ReverseDiffusionPredictor, LangevinCorrector


def train(config, logdir):
    n_dev = jax.local_device_count()
    assert config.training.batch_size % n_dev == 0
    local_bs = config.training.batch_size // n_dev
    H, C = config.data.image_size, config.data.num_channels
    ckpt_dir = f'{logdir}/ckpt'

    writer = tf.summary.create_file_writer(logdir)
    rng = jax.random.PRNGKey(0)

    train_ds, _ = get_dataset(config)
    scaler = get_data_scaler(config.data.centered)
    inverse_scaler = get_data_inverse_scaler(config.data.centered)
    train_iter = iter(train_ds)

    sde, eps = get_sde(config)
    model = UNet(config=config)
    rng, init_rng = jax.random.split(rng)
    params = model.init(init_rng, jnp.ones((1, H, H, C)), jnp.ones(1))

    lr = config.training.learning_rate
    schedule = optax.join_schedules(
        [optax.linear_schedule(0.0, lr, config.training.warmup),
         optax.constant_schedule(lr)],
        boundaries=[config.training.warmup],
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.training.grad_clip),
        optax.adam(learning_rate=schedule),
    )
    opt_state = optimizer.init(params)
    ema_params = params
    start_step = 1

    pathlib.Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    ckpt_mgr = ocp.CheckpointManager(
        ckpt_dir, ocp.PyTreeCheckpointer(),
        options=ocp.CheckpointManagerOptions(max_to_keep=2, create=True),
    )
    if ckpt_mgr.latest_step() is not None:
        target = {'params': params, 'ema_params': ema_params,
                  'opt_state': opt_state, 'step': 0}
        r = ckpt_mgr.restore(ckpt_mgr.latest_step(), items=target)
        params, ema_params, opt_state = r['params'], r['ema_params'], r['opt_state']
        start_step = int(r['step']) + 1
        print(f"Resumed from step {start_step - 1}")

    rep  = lambda x: jax.device_put_replicated(x, jax.local_devices())
    unre = lambda x: jax.tree.map(lambda a: a[0], x)
    params = rep(params); ema_params = rep(ema_params); opt_state = rep(opt_state)

    loss_fn = get_loss_fn(sde, model, train=True,
                          reduce_mean=config.training.reduce_mean,
                          continuous=config.training.continuous)

    @functools.partial(jax.pmap, axis_name='batch')
    def train_step(rng, params, opt_state, batch):
        loss, grads = jax.value_and_grad(lambda p: loss_fn(rng, p, batch))(params)
        grads = jax.lax.pmean(grads, 'batch')
        loss  = jax.lax.pmean(loss,  'batch')
        updates, new_opt = optimizer.update(grads, opt_state, params)
        return loss, optax.apply_updates(params, updates), new_opt

    N_SAMPLES = 4
    SNR = getattr(config.training, 'snr', 0.16)

    @jax.jit
    def pc_sample(rng, params):
        score_fn = get_score_fn(sde, model, params, continuous=config.training.continuous)
        # VE (discrete + continuous): reverse-diffusion predictor (reference for both)
        # VP / sub-VP: Euler-Maruyama
        if isinstance(sde, VESDE):
            predictor = ReverseDiffusionPredictor(sde, score_fn)
        else:
            predictor = EulerMaruyamaPredictor(sde, score_fn)
        corrector = LangevinCorrector(sde, score_fn, SNR, 1)

        rng, init_rng = jax.random.split(rng)
        x = sde.prior_sampling(init_rng, (N_SAMPLES, H, H, C))
        timesteps = jnp.linspace(sde.T, eps, sde.N)

        def step_fn(i, val):
            rng, x, x_mean = val
            t = jnp.full((N_SAMPLES,), timesteps[i])
            rng, crng = jax.random.split(rng)
            x, x_mean = corrector.update_fn(crng, x, t)
            rng, prng = jax.random.split(rng)
            x, x_mean = predictor.update_fn(prng, x, t)
            return rng, x, x_mean

        _, _, x_mean = jax.lax.fori_loop(0, sde.N, step_fn, (rng, x, x))
        return jnp.clip(inverse_scaler(x_mean), 0.0, 1.0)

    ema_decay = config.model.ema_rate
    last_sample_t = time.time()
    print(f"{config.training.sde} continuous={config.training.continuous} | "
          f"{n_dev} device(s) | {config.training.n_iters:,} steps | {logdir}")

    for step in range(start_step, config.training.n_iters + 1):
        batch = jnp.array(scaler(next(train_iter).numpy())).reshape(n_dev, local_bs, H, H, C)
        rng, *step_rngs = jax.random.split(rng, n_dev + 1)

        loss, params, opt_state = train_step(jnp.array(step_rngs), params, opt_state, batch)
        ema_params = jax.tree.map(
            lambda e, p: ema_decay * e + (1 - ema_decay) * p, ema_params, params)

        with writer.as_default():
            tf.summary.scalar('train/loss', float(loss[0]), step=step)
        if step % 500 == 0:
            print(f"  step {step:7d}  loss {float(loss[0]):.4f}")

        if time.time() - last_sample_t >= 30 * 60:
            print(f"  [step {step}] sampling...")
            rng, srng = jax.random.split(rng)
            imgs = np.array(pc_sample(srng, unre(ema_params)))
            grid = np.concatenate([np.concatenate([imgs[0], imgs[1]], 1),
                                   np.concatenate([imgs[2], imgs[3]], 1)], 0)[None]
            with writer.as_default():
                tf.summary.image('samples/pc_4', grid, step=step)
            writer.flush()
            last_sample_t = time.time()

        if step % 10_000 == 0:
            ckpt_mgr.save(step, items={
                'params': unre(params), 'ema_params': unre(ema_params),
                'opt_state': unre(opt_state), 'step': step,
            })

    writer.flush()
    print("Done.")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True,
                   help=f'Config name. Available: {sorted(__import__("config").CONFIGS)}')
    p.add_argument('--logdir', required=True)
    p.add_argument('--n_iters', type=int, default=None,
                   help='Override config n_iters (useful for short continuation runs)')
    args = p.parse_args()
    cfg = get_config(args.config)
    if args.n_iters is not None:
        cfg.training.n_iters = args.n_iters
    train(cfg, args.logdir)
