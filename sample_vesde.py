#!/usr/bin/env python
"""
sample_vesde.py — FID sweep: samplers × step configs on vesde_cifar10.

Samplers    : ancestral, rev_diffusion, prob_flow
Step configs: P1000, P2000, C2000, PC1000

Results (FID) → assets/samples/vesde_sweep/results.txt
"""

import pathlib, time
import numpy as np
import jax, jax.numpy as jnp
import orbax.checkpoint as ocp
import tensorflow as tf
import tensorflow_hub as tfhub

from config import get_config
from sde import VESDE
from model import UNet
from score import get_score_fn
from datasets import get_data_inverse_scaler, get_dataset
from samplers import (
    AncestralSamplingPredictor, ReverseDiffusionPredictor,
    EulerMaruyamaPredictor, LangevinCorrector, Predictor, Corrector,
)

INCEPTION_URL = 'https://tfhub.dev/tensorflow/tfgan/eval/inception/1'

# ── Config ─────────────────────────────────────────────────────────────────────
CKPT   = 'runs/vesde_cifar10/ckpt/190000'
N_SAMP = 10_000
BATCH  = 1024
SNR    = 0.16
EPS    = 1e-5
OUT    = pathlib.Path('assets/samples/vesde_sweep')

# (tag, sampler_N, n_corr_steps, use_predictor)
STEP_CONFIGS = [
    ('P1000',  1000, 0, True),
    ('P2000',  2000, 0, True),
    ('C2000',  1000, 2, False),
    ('PC1000', 1000, 1, True),
]

PREDICTORS = {
    'ancestral':     lambda sde, sf: AncestralSamplingPredictor(sde, sf),
    'rev_diffusion': lambda sde, sf: ReverseDiffusionPredictor(sde, sf, False),
    'prob_flow':     lambda sde, sf: EulerMaruyamaPredictor(sde, sf, probability_flow=True),
}


# ── Inception / FID ────────────────────────────────────────────────────────────

@tf.function
def _inception_batch(x_uint8, model):
    x = (tf.cast(x_uint8, tf.float32) - 127.5) / 127.5
    return model(x)

def run_inception(imgs_uint8, model, bs=500):
    pools = []
    for i in range(0, len(imgs_uint8), bs):
        out = _inception_batch(tf.constant(imgs_uint8[i:i+bs]), model)
        pools.append(out['pool_3'].numpy().reshape(len(out['pool_3']), -1))
    return np.concatenate(pools, 0)

def _sym_sqrtm(m):
    s, u = np.linalg.eigh(m)
    s = np.sqrt(np.maximum(s, 0.0))
    return (u * s) @ u.T

def fid(real_pool, gen_pool):
    r, g = real_pool.astype(np.float64), gen_pool.astype(np.float64)
    mu_r, mu_g = r.mean(0), g.mean(0)
    sr, sg = np.cov(r, rowvar=False), np.cov(g, rowvar=False)
    sqr = _sym_sqrtm(sr)
    tr  = np.trace(_sym_sqrtm(sqr @ sg @ sqr))
    d   = mu_r - mu_g
    return float(d @ d + np.trace(sr) + np.trace(sg) - 2.0 * tr)

def to_uint8(imgs):
    return (np.clip(imgs, 0.0, 1.0) * 255).astype(np.uint8)

def get_real_pool(config, inception_model):
    path = pathlib.Path('assets/stats/cifar10_stats.npz')
    if path.exists():
        return np.load(path)['pool_3'].reshape(-1, 2048)
    print("  Building real stats from training data …")
    path.parent.mkdir(parents=True, exist_ok=True)
    train_ds, _ = get_dataset(config)
    pools, total = [], 0
    for batch in train_ds:
        pools.append(run_inception(to_uint8(np.array(batch)), inception_model))
        total += len(pools[-1])
        if total >= 50_000:
            break
    real = np.concatenate(pools)[:50_000]
    np.savez_compressed(path, pool_3=real)
    return real


# ── Sampling ───────────────────────────────────────────────────────────────────

def load_model(ckpt_path, config):
    model = UNet(config=config)
    path  = str(pathlib.Path(ckpt_path).resolve() / 'default')
    params = ocp.PyTreeCheckpointer().restore(path)['ema_params']
    return model, jax.device_put(params, jax.devices()[0])

def make_sampler(sde, shape, pred, corr, inv_scaler):
    @jax.jit
    def _sample(rng):
        rng, k = jax.random.split(rng)
        x  = sde.prior_sampling(k, shape)
        ts = jnp.linspace(sde.T, EPS, sde.N)

        def step(i, val):
            rng, x, xm = val
            t = jnp.full((shape[0],), ts[i])
            rng, k = jax.random.split(rng)
            x, xm = corr.update_fn(k, x, t)
            rng, k = jax.random.split(rng)
            x, xm = pred.update_fn(k, x, t)
            return rng, x, xm

        _, _, xm = jax.lax.fori_loop(0, sde.N, step, (rng, x, x))
        return inv_scaler(xm)
    return _sample

def generate_pool(sampler, rng, inception_model):
    """Generate N_SAMP images and return Inception pool_3 activations."""
    n_batches = (N_SAMP + BATCH - 1) // BATCH
    pools = []
    t0 = time.time()
    for i in range(n_batches):
        rng, k = jax.random.split(rng)
        imgs = to_uint8(np.clip(np.array(sampler(k)), 0, 1))
        pools.append(run_inception(imgs, inception_model))
        print(f"    batch {i+1}/{n_batches}  {time.time()-t0:.0f}s", flush=True)
    return np.concatenate(pools)[:N_SAMP], rng


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results_file = OUT / 'results.txt'
    rng = jax.random.PRNGKey(42)

    config     = get_config('vesde_ddpm_disc')
    H, C       = config.data.image_size, config.data.num_channels
    inv_scaler = get_data_inverse_scaler(config.data.centered)
    sigma_min  = config.model.sigma_min
    sigma_max  = config.model.sigma_max

    print("Loading Inception …")
    inception = tfhub.load(INCEPTION_URL)
    real_pool = get_real_pool(config, inception)
    print(f"  real pool_3: {real_pool.shape}")

    print(f"\nLoading checkpoint: {CKPT}")
    model, params = load_model(CKPT, config)

    # Score SDE fixed at N=1000 — discrete model trained with indices [0, 999]
    score_sde = VESDE(sigma_min, sigma_max, 1000)
    score_fn  = get_score_fn(score_sde, model, params,
                             train=False, continuous=False)

    results = {}
    # Load any previously saved results
    if results_file.exists():
        for line in results_file.read_text().splitlines():
            parts = line.split()
            if len(parts) == 3:
                results[parts[0]] = (float(parts[1]), float(parts[2]))

    total = len(PREDICTORS) * len(STEP_CONFIGS)
    idx   = 0

    for pred_name, pred_fn in PREDICTORS.items():
        for tag, N, n_corr, use_pred in STEP_CONFIGS:
            idx += 1
            run_name = f"{pred_name}_{tag}"

            if run_name in results:
                fid_val, elapsed = results[run_name]
                print(f"[{idx}/{total}] {run_name}  FID={fid_val:.3f}  (cached)")
                continue

            print(f"\n[{idx}/{total}] {run_name}  N={N}  n_corr={n_corr}")

            sde  = VESDE(sigma_min, sigma_max, N)
            pred = pred_fn(sde, score_fn) if use_pred else Predictor()
            corr = (LangevinCorrector(sde, score_fn, SNR, n_corr)
                    if n_corr > 0 else Corrector())

            sampler = make_sampler(sde, (BATCH, H, H, C), pred, corr, inv_scaler)

            rng, k = jax.random.split(rng)
            sampler(k).block_until_ready()      # JIT warm-up
            print("  compiled — generating …")

            t0 = time.time()
            gen_pool, rng = generate_pool(sampler, rng, inception)
            elapsed = time.time() - t0

            fid_val = fid(real_pool, gen_pool)
            results[run_name] = (fid_val, elapsed)
            print(f"  FID={fid_val:.3f}  ({elapsed:.0f}s)")

            # Append to results file immediately
            with open(results_file, 'a') as f:
                f.write(f"{run_name}  {fid_val:.4f}  {elapsed:.0f}\n")

    print(f"\n{'─'*45}")
    print(f"{'run':<25}  {'FID':>8}  {'time(s)':>8}")
    print(f"{'─'*45}")
    for name, (fid_val, elapsed) in sorted(results.items()):
        print(f"{name:<25}  {fid_val:8.3f}  {elapsed:8.0f}")
    print(f"{'─'*45}")
    print(f"Results saved → {results_file}")


if __name__ == '__main__':
    main()
