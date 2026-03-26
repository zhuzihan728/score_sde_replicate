#!/usr/bin/env python
"""
eval.py — benchmark all trained models.

CIFAR models : 10k samples → FID + IS → save 8×8 grid
CelebA model : 8×8 grid only (no FID/IS)

Results saved to assets/eval/results.txt
"""

import pathlib, time
import numpy as np
import jax, jax.numpy as jnp
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import orbax.checkpoint as ocp
import tensorflow as tf
import tensorflow_hub as tfhub

from config import get_config
from sde import get_sde
from model import UNet
from score import get_score_fn
from datasets import get_data_inverse_scaler

from samplers import (
    EulerMaruyamaPredictor, ReverseDiffusionPredictor, AncestralSamplingPredictor,
    LangevinCorrector, Corrector, Predictor,
)

# ── Constants ──────────────────────────────────────────────────────────────────
INCEPTION_TFHUB = 'https://tfhub.dev/tensorflow/tfgan/eval/inception/1'
N_SAMPLES  = 10_000
BATCH_SIZE = 1024
GRID_N     = 64
OUT_ROOT   = pathlib.Path('assets/eval')

# ── Model registry ─────────────────────────────────────────────────────────────
# (display_name, config_name, ckpt_path, celeba_only)
MODELS = [
    ('vpsde_disc',    'vpsde_ddpm_disc',         'runs/ddpm_cifar10_low/ckpt/190000',       False),
    ('vpsde_cont',    'vpsde_ddpm_cont',          'runs/ddpm_cont_cifar10_1gpu/ckpt/190000', False),
    ('subvpsde_cont', 'subvpsde_ddpm_cont',       'runs/subvp_cont_cifar10/ckpt/190000',     False),
    ('vesde_disc',    'vesde_ddpm_disc',           'runs/vesde_cifar10/ckpt/190000',          False),
    ('vesde_cont',    'vesde_ddpm_cont',           'runs/vesde_cont_cifar10/ckpt/190000',     False),
    ('ddpmpp_cont',   'vpsde_ddpmpp_cont',         'runs/ddpmpp_cont_cifar10/ckpt/160000',    False),
    ('ncsnpp_cont',   'vesde_ncsnpp_cont',         'runs/ncsnpp_cont_cifar10/ckpt/160000',    False),
    ('ncsnpp_celeba', 'vesde_ncsnpp_celeba_disc',  'runs/ncsnpp_celeba/ckpt/50000',           True),
]

# Best PC per (sde, continuous) — reverse_diffusion for all per paper Table 1
BEST_PC = {
    ('vesde',    True):  ('reverse_diffusion',  'langevin', 0.16),
    ('vesde',    False): ('reverse_diffusion',  'langevin', 0.16),
    ('vpsde',    False): ('ancestral_sampling', 'none',     0.16),
    ('vpsde',    True):  ('euler_maruyama',     'none',     0.16),
    ('subvpsde', False): ('ancestral_sampling', 'none',     0.16),
    ('subvpsde', True):  ('euler_maruyama',     'none',     0.16),
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_ckpt(ckpt_path, config):
    model = UNet(config=config)
    default_path = str(pathlib.Path(ckpt_path).resolve() / 'default')
    checkpointer = ocp.PyTreeCheckpointer()
    restored = checkpointer.restore(default_path)
    ema_params = jax.device_put(restored['ema_params'], jax.devices()[0])
    return model, ema_params


def build_predictor(pred_type, sde, score_fn):
    if pred_type == 'reverse_diffusion':
        return ReverseDiffusionPredictor(sde, score_fn, False)
    if pred_type == 'euler_maruyama':
        return EulerMaruyamaPredictor(sde, score_fn, False)
    if pred_type == 'ancestral_sampling':
        return AncestralSamplingPredictor(sde, score_fn)
    return Predictor()


def build_corrector(corr_type, sde, score_fn, snr):
    if corr_type == 'langevin':
        return LangevinCorrector(sde, score_fn, snr, n_steps=1)
    return Corrector()


def make_sampler(sde, shape, pred, corr, inverse_scaler, eps):
    """JIT-compiled PC sampler using fori_loop."""
    n_corr = getattr(corr, 'n_steps', 0) or 0

    @jax.jit
    def _sample(rng):
        rng, k = jax.random.split(rng)
        x = sde.prior_sampling(k, shape)
        ts = jnp.linspace(sde.T, eps, sde.N)

        def step(i, val):
            rng, x, x_mean = val
            t = jnp.full((shape[0],), ts[i])
            rng, k = jax.random.split(rng)
            x, x_mean = corr.update_fn(k, x, t)
            rng, k = jax.random.split(rng)
            x, x_mean = pred.update_fn(k, x, t)
            return rng, x, x_mean

        _, x, x_mean = jax.lax.fori_loop(0, sde.N, step, (rng, x, x))
        return inverse_scaler(x_mean), sde.N * (n_corr + 1)

    return _sample


def generate_samples(sampler_fn, n_samples, batch_size, rng):
    all_imgs = []
    n_batches = int(np.ceil(n_samples / batch_size))
    t0 = time.time()
    for i in range(n_batches):
        rng, k = jax.random.split(rng)
        imgs, _ = sampler_fn(k)
        all_imgs.append(np.clip(np.array(imgs), 0.0, 1.0))
        print(f"    batch {i+1}/{n_batches}  ({time.time()-t0:.0f}s)")
    return np.concatenate(all_imgs, axis=0)[:n_samples]


def save_grid(imgs, path, nrow=8):
    n, H, W, C = imgs.shape
    nrows = int(np.ceil(n / nrow))
    canvas = np.ones((nrows * H, nrow * W, C), dtype=np.float32)
    for idx, img in enumerate(imgs):
        r, c = divmod(idx, nrow)
        canvas[r*H:(r+1)*H, c*W:(c+1)*W] = np.clip(img, 0, 1)
    plt.imsave(str(path), canvas)
    print(f"  Saved grid → {path}")


def to_uint8(imgs):
    return (np.clip(imgs, 0.0, 1.0) * 255).astype(np.uint8)


# ── Inception / FID / IS ───────────────────────────────────────────────────────

def get_inception_model():
    return tfhub.load(INCEPTION_TFHUB)


@tf.function
def _run_inception_batch(batch_uint8, model):
    x = (tf.cast(batch_uint8, tf.float32) - 127.5) / 127.5
    return model(x)


def run_inception(samples_uint8, model, batch_size=500):
    pools, logits_all = [], []
    for i in range(0, len(samples_uint8), batch_size):
        out = _run_inception_batch(tf.constant(samples_uint8[i:i+batch_size]), model)
        pools.append(out['pool_3'].numpy().reshape(len(out['pool_3']), -1))
        logits_all.append(out['logits'].numpy())
    return np.concatenate(pools, 0), np.concatenate(logits_all, 0)


def inception_score(logits):
    logits = logits.astype(np.float64)
    log_p = logits - np.log(
        np.sum(np.exp(logits - logits.max(1, keepdims=True)), axis=1, keepdims=True)
    ) - logits.max(1, keepdims=True)
    p = np.exp(log_p)
    kl = (p * (log_p - np.log(p.mean(0)))).sum(1)
    return float(np.exp(kl.mean()))


def _sym_sqrtm(m, eps=1e-10):
    s, u = np.linalg.eigh(m)
    s = np.where(s < eps, 0.0, np.sqrt(np.maximum(s, 0.0)))
    return (u * s) @ u.T


def frechet_distance(act_real, act_gen):
    act_real = act_real.astype(np.float64)
    act_gen  = act_gen.astype(np.float64)
    mu_r, mu_g   = act_real.mean(0), act_gen.mean(0)
    sigma_r      = np.cov(act_real, rowvar=False)
    sigma_g      = np.cov(act_gen,  rowvar=False)
    sqrt_r       = _sym_sqrtm(sigma_r)
    sqrt_trace   = np.trace(_sym_sqrtm(sqrt_r @ sigma_g @ sqrt_r))
    diff         = mu_r - mu_g
    return float(diff @ diff + np.trace(sigma_r) + np.trace(sigma_g) - 2.0 * sqrt_trace)


def load_real_stats(config, inception_model):
    dataset = config.data.dataset.lower()
    stats_path = pathlib.Path(f'assets/stats/{dataset}_stats.npz')
    if stats_path.exists():
        return np.load(stats_path)['pool_3'].reshape(-1, 2048)
    print(f"  Building real stats from training set ({stats_path}) …")
    from datasets import get_dataset
    pathlib.Path('assets/stats').mkdir(parents=True, exist_ok=True)
    train_ds, _ = get_dataset(config)
    pools = []
    total = 0
    for i, batch in enumerate(train_ds):
        pool = run_inception(to_uint8(np.array(batch)), inception_model)[0]
        pools.append(pool)
        total += len(pool)
        if (i + 1) % 20 == 0:
            print(f"    real stats batch {i+1}  ({total} images)")
        if total >= 50_000:
            break
    real_pool3 = np.concatenate(pools, 0)[:50_000]
    np.savez_compressed(stats_path, pool_3=real_pool3)
    print(f"  Saved {stats_path}")
    return real_pool3


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rng = jax.random.PRNGKey(0)

    inception_model = None
    real_pool3      = None
    results = []

    for name, cfg_name, ckpt_path, celeba_only in MODELS:
        print(f"\n{'='*60}")
        print(f"Model: {name}  ({ckpt_path})")

        out_dir = OUT_ROOT / name
        out_dir.mkdir(parents=True, exist_ok=True)

        result_file = out_dir / 'result.txt'
        if result_file.exists():
            parts = result_file.read_text().split()
            IS, FID, elapsed = float(parts[1]), float(parts[2]), float(parts[3])
            print(f"  Already done — IS={IS:.4f}  FID={FID:.4f}  (skipping)")
            results.append((name, IS, FID, elapsed))
            continue

        config = get_config(cfg_name)
        H, C   = config.data.image_size, config.data.num_channels
        sde_obj, eps = get_sde(config)
        inverse_scaler = get_data_inverse_scaler(config.data.centered)

        sde_type = config.training.sde
        cont     = config.training.continuous
        pred_type, corr_type, snr = BEST_PC.get(
            (sde_type, cont), ('reverse_diffusion', 'none', 0.16)
        )
        if celeba_only:
            snr = 0.17
        print(f"  PC: predictor={pred_type}  corrector={corr_type}  snr={snr}")

        print(f"  Loading {ckpt_path} …")
        model, ema_params = load_ckpt(ckpt_path, config)
        score_fn = get_score_fn(sde_obj, model, ema_params,
                                train=False, continuous=cont)

        pred  = build_predictor(pred_type, sde_obj, score_fn)
        corr  = build_corrector(corr_type, sde_obj, score_fn, snr)

        if celeba_only:
            # ── CelebA: 8×8 grid only ─────────────────────────────────────────
            print("  Generating 8×8 grid (CelebA) …")
            grid_sampler = make_sampler(sde_obj, (GRID_N, H, H, C), pred, corr, inverse_scaler, eps)
            rng, k = jax.random.split(rng)
            grid_imgs, _ = grid_sampler(k)
            save_grid(np.clip(np.array(grid_imgs), 0.0, 1.0), out_dir / 'grid.png', nrow=8)
            print("  CelebA — skipping FID/IS.")
            continue

        # ── Lazy-load Inception + real stats on first CIFAR model ────────────
        if inception_model is None:
            print("Loading InceptionV1 …")
            inception_model = get_inception_model()
            cifar_config = get_config('vpsde_ddpm_disc')
            real_pool3   = load_real_stats(cifar_config, inception_model)
            print(f"  Real pool_3: {real_pool3.shape}")

        # ── 10k samples → FID + IS + grid (single JIT) ───────────────────────
        stats_file = out_dir / 'statistics.npz'
        elapsed = 0
        if stats_file.exists():
            print(f"  Found cached statistics, skipping sampling …")
            cached = np.load(stats_file)
            gen_pool3  = cached['pool_3'].reshape(-1, 2048)
            gen_logits = cached['logits']
        else:
            print(f"  Generating {N_SAMPLES} samples (batch={BATCH_SIZE}) …")
            batch_shape  = (BATCH_SIZE, H, H, C)
            bulk_sampler = make_sampler(sde_obj, batch_shape, pred, corr, inverse_scaler, eps)
            t0 = time.time()
            samples = generate_samples(bulk_sampler, N_SAMPLES, BATCH_SIZE, rng)
            elapsed = time.time() - t0
            print(f"  Generated in {elapsed:.0f}s")

            # reuse first GRID_N samples for the grid (no extra JIT)
            save_grid(samples[:GRID_N], out_dir / 'grid.png', nrow=8)

            print("  Running Inception …")
            gen_pool3, gen_logits = run_inception(to_uint8(samples), inception_model)
            np.savez_compressed(stats_file, pool_3=gen_pool3, logits=gen_logits)

        IS  = inception_score(gen_logits)
        FID = frechet_distance(real_pool3, gen_pool3)
        print(f"  IS={IS:.4f}  FID={FID:.4f}")

        results.append((name, IS, FID, elapsed))

        # save per-model result immediately
        (out_dir / 'result.txt').write_text(
            f"{name}  {IS:.4f}  {FID:.4f}  {elapsed:.0f}\n"
        )


if __name__ == '__main__':
    main()
