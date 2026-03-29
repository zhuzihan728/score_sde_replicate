#!/usr/bin/env python
"""
sample_ncsnpp_celeba_nfe.py — ODE NFE-vs-quality sweep on ncsnpp_celeba.

Fix N_IMAGES noise vectors, then run probability-flow ODE with progressively
tighter tolerances.  Because the starting point z is identical for every run,
each column of the output grid shows the same face at increasing quality.

Checkpoint : runs/ncsnpp_celeba/ckpt/50000
Config     : vesde_ncsnpp_celeba_disc
Outputs    : assets/samples/ncsnpp_celeba-nfe/
                nfe_sweep.npy      (n_tol, N_IMAGES, H, H, C)  float32
                nfe_sweep_grid.png  rows = images, cols = tolerances (low→high)
                nfe_log.txt         actual NFE recorded per tolerance level
"""

import pathlib
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
from samplers import ode_sampler

CKPT      = 'runs/ncsnpp_celeba/ckpt/50000'
CONFIG    = 'vesde_ncsnpp_celeba_disc'
N_IMAGES  = 4
SEED      = 42
OUT       = pathlib.Path('assets/samples/ncsnpp_celeba-nfe')

# Tolerance sweep: loose → tight  (lower tol = more NFE = better quality)
TOL_CONFIGS = [
    ('tol=1e-1', 1e-1, 1e-1),
    ('tol=1e-2', 1e-2, 1e-2),
    ('tol=1e-3', 1e-3, 1e-3),
    ('tol=1e-4', 1e-4, 1e-4),
    ('tol=1e-5', 1e-5, 1e-5),   # reference quality
]


def load_ckpt(ckpt_path, config):
    model = UNet(config=config)
    default_path = str(pathlib.Path(ckpt_path).resolve() / 'default')
    restored = ocp.PyTreeCheckpointer().restore(default_path)
    ema_params = jax.device_put(restored['ema_params'], jax.devices()[0])
    return model, ema_params


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    config = get_config(CONFIG)
    H, C   = config.data.image_size, config.data.num_channels
    sde, eps = get_sde(config)
    inverse_scaler = get_data_inverse_scaler(config.data.centered)

    print(f"Loading {CKPT} …")
    model, ema_params = load_ckpt(CKPT, config)
    score_fn = get_score_fn(sde, model, ema_params,
                            train=False, continuous=config.training.continuous)

    shape = (N_IMAGES, H, H, C)

    # Fix the starting noise — same z for every tolerance level
    rng = jax.random.PRNGKey(SEED)
    rng, k = jax.random.split(rng)
    z = sde.prior_sampling(k, shape)
    print(f"Fixed prior noise z: shape={z.shape}")

    all_imgs = []   # (n_tol, N_IMAGES, H, H, C)
    nfe_log  = []

    for label, rtol, atol in TOL_CONFIGS:
        print(f"\nODE  {label}  rtol={rtol}  atol={atol} …", flush=True)
        sampler_fn = ode_sampler(sde, score_fn, shape, inverse_scaler,
                                 rtol=rtol, atol=atol, eps=eps)
        rng, k = jax.random.split(rng)
        imgs, nfe = sampler_fn(k, z=z)
        imgs = np.clip(np.array(imgs).reshape(N_IMAGES, H, H, C), 0.0, 1.0).astype(np.float32)
        all_imgs.append(imgs)
        nfe_log.append((label, nfe))
        print(f"  NFE={nfe}  done.", flush=True)

    all_imgs = np.stack(all_imgs, axis=0)   # (n_tol, N_IMAGES, H, H, C)
    np.save(str(OUT / 'nfe_sweep.npy'), all_imgs)
    print(f"\nSaved {OUT}/nfe_sweep.npy  shape={all_imgs.shape}")

    # ── NFE log ───────────────────────────────────────────────────────────────
    log_lines = [f"{lbl}  NFE={nfe}" for lbl, nfe in nfe_log]
    (OUT / 'nfe_log.txt').write_text('\n'.join(log_lines) + '\n')
    print('\n'.join(log_lines))

    # ── Grid PNG: rows = images, cols = tolerance levels ─────────────────────
    n_tol = len(TOL_CONFIGS)
    fig, axes = plt.subplots(N_IMAGES, n_tol,
                             figsize=(n_tol * 2, N_IMAGES * 2))
    for col, (label, rtol, atol) in enumerate(TOL_CONFIGS):
        nfe_val = nfe_log[col][1]
        axes[0, col].set_title(f"{label}\nNFE={nfe_val}", fontsize=7)
        for row in range(N_IMAGES):
            ax = axes[row, col]
            img = all_imgs[col, row]
            ax.imshow(img if C == 3 else img[..., 0], cmap=None if C == 3 else 'gray')
            ax.axis('off')

    plt.tight_layout()
    grid_path = OUT / 'nfe_sweep_grid.png'
    plt.savefig(str(grid_path), dpi=150, bbox_inches='tight')
    print(f"Saved {grid_path}")


if __name__ == '__main__':
    main()
