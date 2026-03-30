#!/usr/bin/env python
"""
prob_flow_nfe.py — Probability flow ODE samples vs NFE on CelebA (NCSN++).
Layout: 2 rows x 3 cols, columns = NFE {14, 86, 548}.
Output: assets/samples/prob_flow_nfe.png
"""
import argparse, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import numpy as np
import jax, jax.numpy as jnp
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import orbax.checkpoint as ocp

from config import get_config
from sde import VESDE
from model import UNet
from score import get_score_fn
from datasets import get_data_inverse_scaler
from samplers import ProbabilityFlowPredictor, Corrector

VARIANTS = {
    'disc': ('vesde_ncsnpp_celeba_disc', 'runs/ncsnpp_celeba_disc/ckpt/50000', False),
    'cont': ('vesde_ncsnpp_celeba_cont', 'runs/ncsnpp_celeba_cont/ckpt/50000', True),
}
NFES   = [14, 86, 548]
N_ROWS = 2
EPS    = 1e-5


def load_model(ckpt_path, config):
    model  = UNet(config=config)
    path   = str(pathlib.Path(ckpt_path).resolve() / 'default')
    params = ocp.PyTreeCheckpointer().restore(path)['ema_params']
    return model, jax.device_put(params, jax.devices()[0])


def make_sampler(sde, shape, pred, inv_scaler):
    corr = Corrector()

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=['disc', 'cont'], default='disc',
                        help='CelebA model variant: disc (default) or cont')
    args = parser.parse_args()

    cfg_name, ckpt, continuous = VARIANTS[args.variant]
    out = pathlib.Path(f'assets/samples/prob_flow_nfe_{args.variant}.png')

    out.parent.mkdir(parents=True, exist_ok=True)
    rng = jax.random.PRNGKey(0)

    config     = get_config(cfg_name)
    H, C       = config.data.image_size, config.data.num_channels
    inv_scaler = get_data_inverse_scaler(config.data.centered)
    sigma_min  = config.model.sigma_min
    sigma_max  = config.model.sigma_max

    print(f"Loading checkpoint: {ckpt}")
    model, params = load_model(ckpt, config)

    score_sde = VESDE(sigma_min, sigma_max, 1000)
    score_fn  = get_score_fn(score_sde, model, params, train=False, continuous=continuous)

    shape = (N_ROWS, H, H, C)
    cols  = []

    for N in NFES:
        print(f"NFE={N} ...", flush=True)
        sde     = VESDE(sigma_min, sigma_max, N)
        pred    = ProbabilityFlowPredictor(sde, score_fn)
        sampler = make_sampler(sde, shape, pred, inv_scaler)

        rng, k = jax.random.split(rng)
        sampler(k).block_until_ready()       # JIT warm-up
        rng, k = jax.random.split(rng)
        imgs = np.clip(np.array(sampler(k)), 0.0, 1.0)
        cols.append(imgs)
        print(f"  done", flush=True)

    fig, axes = plt.subplots(N_ROWS, len(NFES), figsize=(len(NFES) * 3, N_ROWS * 3))
    for col, (N, imgs) in enumerate(zip(NFES, cols)):
        for row in range(N_ROWS):
            ax = axes[row, col]
            ax.imshow(imgs[row])
            ax.axis('off')
            if row == 0:
                ax.set_title(f'NFE = {N}', fontsize=12)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved -> {out}")


if __name__ == '__main__':
    main()
