#!/usr/bin/env python
"""
sample_ncsnpp_celeba.py — Generate 8 CelebA samples via PC (best) and ODE (prob flow).

Best PC (from score_sde paper, vesde discrete):
    predictor : reverse_diffusion
    corrector : langevin  snr=0.17

Checkpoint : runs/ncsnpp_celeba/ckpt/50000
Config     : vesde_ncsnpp_celeba_disc
Outputs    : assets/samples/ncsnpp_celeba-50000/pc_samples.npy   (8, H, H, C) float32
             assets/samples/ncsnpp_celeba-50000/ode_samples.npy  (8, H, H, C) float32
"""

import pathlib
import numpy as np
import jax, jax.numpy as jnp
import orbax.checkpoint as ocp

from config import get_config
from sde import get_sde
from model import UNet
from score import get_score_fn
from datasets import get_data_inverse_scaler
from samplers import (
    ReverseDiffusionPredictor, LangevinCorrector,
    pc_sampler, ode_sampler,
)

CKPT   = 'runs/ncsnpp_celeba/ckpt/50000'
CONFIG = 'vesde_ncsnpp_celeba_disc'
N      = 8
SNR    = 0.17   # CelebA: 0.17 per score_sde/configs/ve/celeba_ncsnpp.py
OUT    = pathlib.Path('assets/samples/ncsnpp_celeba-50000')
SEED   = 42


def load_ckpt(ckpt_path, config):
    H, C = config.data.image_size, config.data.num_channels
    model = UNet(config=config)
    model.init(jax.random.PRNGKey(0), jnp.ones((1, H, H, C)), jnp.ones(1))
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

    shape = (N, H, H, C)
    rng   = jax.random.PRNGKey(SEED)

    # ── 1. Best PC: reverse_diffusion predictor + langevin corrector ───────────
    print(f"PC sampling (reverse_diffusion + langevin, snr={SNR}) …")
    pred  = ReverseDiffusionPredictor(sde, score_fn, False)
    corr  = LangevinCorrector(sde, score_fn, SNR, n_steps=1)
    pc_fn = pc_sampler(sde, shape, pred, corr, inverse_scaler, denoise=True, epsilon=eps)
    rng, k = jax.random.split(rng)
    pc_imgs, _ = pc_fn(k)
    pc_imgs = np.clip(np.array(pc_imgs), 0.0, 1.0).astype(np.float32)
    np.save(str(OUT / 'pc_samples.npy'), pc_imgs)
    print(f"  Saved {OUT}/pc_samples.npy  shape={pc_imgs.shape}")

    # ── 2. ODE: probability flow ───────────────────────────────────────────────
    print("ODE sampling (probability flow) …")
    ode_fn = ode_sampler(sde, score_fn, shape, inverse_scaler, eps=eps)
    rng, k = jax.random.split(rng)
    ode_imgs, nfe = ode_fn(k)
    ode_imgs = np.clip(np.array(ode_imgs), 0.0, 1.0).astype(np.float32)
    np.save(str(OUT / 'ode_samples.npy'), ode_imgs)
    print(f"  NFE: {nfe}")
    print(f"  Saved {OUT}/ode_samples.npy  shape={ode_imgs.shape}")


if __name__ == '__main__':
    main()
