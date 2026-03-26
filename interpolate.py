#!/usr/bin/env python
"""
For each pair of images:
  1. Sample two latent codes z1, z2 from the prior (VESDE Gaussian).
  2. Spherically interpolate (slerp) between them for N_INTERP steps.
  3. Decode each interpolated latent via the probability-flow ODE.
"""

import argparse, pathlib
import numpy as np
import jax, jax.numpy as jnp
import orbax.checkpoint as ocp
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import get_config
from sde import get_sde
from model import UNet
from score import get_score_fn
from datasets import get_data_inverse_scaler
from samplers import ode_sampler, ReverseDiffusionPredictor, LangevinCorrector

# ── Config (mirrors eval.py / sample.py) ──────────────────────────────────────
CFG_NAME  = 'vesde_ncsnpp_celeba_disc'
CKPT_PATH = 'runs/ncsnpp_celeba/ckpt/50000'
SNR       = 0.17   # CelebA override from eval.py

OUT_DIR   = pathlib.Path('assets/eval/ncsnpp_celeba/interpolation')

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_ckpt(ckpt_path, config):
    model = UNet(config=config)
    default_path = str(pathlib.Path(ckpt_path).resolve() / 'default')
    checkpointer = ocp.PyTreeCheckpointer()
    restored = checkpointer.restore(default_path)
    ema_params = jax.device_put(restored['ema_params'], jax.devices()[0])
    return model, ema_params


def slerp(z1, z2, alpha):
    """Spherical linear interpolation between two flat vectors."""
    z1_flat = z1.ravel()
    z2_flat = z2.ravel()
    omega = jnp.arccos(
        jnp.clip(
            jnp.dot(z1_flat, z2_flat) /
            (jnp.linalg.norm(z1_flat) * jnp.linalg.norm(z2_flat) + 1e-8),
            -1.0, 1.0,
        )
    )
    sin_omega = jnp.sin(omega) + 1e-8
    coeff1 = jnp.sin((1.0 - alpha) * omega) / sin_omega
    coeff2 = jnp.sin(alpha * omega) / sin_omega
    return (coeff1 * z1_flat + coeff2 * z2_flat).reshape(z1.shape)


def save_grid(imgs, path, nrow):
    """Save float32 [0,1] (N,H,W,C) images as a single-row PNG grid."""
    n, H, W, C = imgs.shape
    nrows = int(np.ceil(n / nrow))
    canvas = np.ones((nrows * H, nrow * W, C), dtype=np.float32)
    for idx, img in enumerate(imgs):
        r, c = divmod(idx, nrow)
        canvas[r*H:(r+1)*H, c*W:(c+1)*W] = np.clip(img, 0, 1)
    plt.imsave(str(path), canvas)
    print(f"  Saved {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_pairs',   type=int, default=4,
                        help='Number of interpolation pairs (rows in the grid)')
    parser.add_argument('--n_interp',  type=int, default=8,
                        help='Number of interpolation steps per pair (including endpoints)')
    parser.add_argument('--seed',      type=int, default=0)
    parser.add_argument('--out_dir',   default=None,
                        help='Output directory (default: assets/eval/ncsnpp_celeba/interpolation)')
    parser.add_argument('--sampler',   choices=['ode', 'pc'], default='ode',
                        help='Decoder to use: ode (probability-flow) or pc (predictor-corrector)')
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir) if args.out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    config = get_config(CFG_NAME)
    H, C   = config.data.image_size, config.data.num_channels
    sde, eps = get_sde(config)
    inverse_scaler = get_data_inverse_scaler(config.data.centered)

    print(f"Loading {CKPT_PATH} …")
    model, ema_params = load_ckpt(CKPT_PATH, config)
    score_fn = get_score_fn(sde, model, ema_params,
                            train=False, continuous=config.training.continuous)

    rng = jax.random.PRNGKey(args.seed)

    # ── Build decoder ─────────────────────────────────────────────────────────
    if args.sampler == 'pc':
        pred = ReverseDiffusionPredictor(sde, score_fn, False)
        corr = LangevinCorrector(sde, score_fn, SNR, n_steps=1)

        @jax.jit
        def pc_from_z(rng, z):
            ts = jnp.linspace(sde.T, eps, sde.N)
            def step(i, val):
                rng, x, x_mean = val
                t = jnp.full((1,), ts[i])
                rng, k = jax.random.split(rng)
                x, x_mean = corr.update_fn(k, x, t)
                rng, k = jax.random.split(rng)
                x, x_mean = pred.update_fn(k, x, t)
                return rng, x, x_mean
            _, _, x_mean = jax.lax.fori_loop(0, sde.N, step, (rng, z, z))
            return inverse_scaler(x_mean)

    # ── Interpolation loop ────────────────────────────────────────────────────
    all_rows = []  # list of (n_interp, H, H, C) arrays

    for pair_idx in range(args.n_pairs):
        print(f"\nPair {pair_idx + 1}/{args.n_pairs}")

        # Sample two independent prior latents
        rng, k1, k2 = jax.random.split(rng, 3)
        z1 = sde.prior_sampling(k1, (1, H, H, C))  # (1, H, H, C)
        z2 = sde.prior_sampling(k2, (1, H, H, C))

        # Build interpolated latent batch (n_interp, H, H, C)
        alphas = np.linspace(0.0, 1.0, args.n_interp)
        z_interp = jnp.stack(
            [slerp(z1[0], z2[0], float(a)) for a in alphas], axis=0
        )  # (n_interp, H, H, C)

        print(f"  Decoding {args.n_interp} latents via {args.sampler.upper()} …")

        if args.sampler == 'ode':
            shape = (args.n_interp, H, H, C)
            decode = ode_sampler(sde, score_fn, shape, inverse_scaler, eps=eps)
            rng, k = jax.random.split(rng)
            imgs, nfe = decode(k, z=z_interp)
            print(f"  NFE: {nfe}")
        else:
            imgs_list = []
            for i in range(args.n_interp):
                z_i = z_interp[i:i+1]  # (1, H, H, C)
                rng, k = jax.random.split(rng)
                img = pc_from_z(k, z_i)
                imgs_list.append(np.clip(np.array(img), 0, 1))
                print(f"    step {i+1}/{args.n_interp}")
            imgs = np.concatenate(imgs_list, axis=0)  # (n_interp, H, H, C)

        imgs = np.clip(np.array(imgs), 0, 1)
        all_rows.append(imgs)

        # Save per-pair strip
        save_grid(imgs, out_dir / f'pair_{pair_idx:02d}.png', nrow=args.n_interp)

    # ── Combined grid (all pairs stacked vertically) ──────────────────────────
    all_imgs = np.concatenate(all_rows, axis=0)  # (n_pairs * n_interp, H, H, C)
    save_grid(all_imgs, out_dir / 'interpolation_grid.png', nrow=args.n_interp)
    print(f"\nDone. Grid: {args.n_pairs} rows × {args.n_interp} cols → {out_dir}/interpolation_grid.png")


if __name__ == '__main__':
    main()
