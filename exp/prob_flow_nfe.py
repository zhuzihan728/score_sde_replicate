import argparse, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
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

VARIANTS = {
    'disc': ('runs/ncsnpp_celeba_disc/ckpt/50000', 'vesde_ncsnpp_celeba_disc'),
    'cont': ('runs/ncsnpp_celeba_cont/ckpt/50000', 'vesde_ncsnpp_celeba_cont'),
}
N_IMAGES = 4
SEED     = 123

TOL_CONFIGS = [
    ('tol=1e-1', 1e-1, 1e-1),
    ('tol=1e-2', 1e-2, 1e-2),
    ('tol=1e-3', 1e-3, 1e-3),
    ('tol=1e-4', 1e-4, 1e-4),
    ('tol=1e-5', 1e-5, 1e-5),
]


def load_ckpt(ckpt_path, config):
    model      = UNet(config=config)
    restored   = ocp.PyTreeCheckpointer().restore(str(pathlib.Path(ckpt_path).resolve() / 'default'))
    ema_params = jax.device_put(restored['ema_params'], jax.devices()[0])
    return model, ema_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=['disc', 'cont'], default='disc')
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()

    ckpt, cfg_name = VARIANTS[args.variant]
    out = pathlib.Path(f'assets/samples/ncsnpp_celeba_nfe_{args.variant}')
    out.mkdir(parents=True, exist_ok=True)

    config = get_config(cfg_name)
    H, C   = config.data.image_size, config.data.num_channels
    sde, eps = get_sde(config)
    inverse_scaler = get_data_inverse_scaler(config.data.centered)

    print(f"Loading {ckpt} …")
    model, ema_params = load_ckpt(ckpt, config)
    score_fn = get_score_fn(sde, model, ema_params, train=False, continuous=config.training.continuous)

    shape = (N_IMAGES, H, H, C)
    rng = jax.random.PRNGKey(args.seed)
    rng, k = jax.random.split(rng)
    z = sde.prior_sampling(k, shape)

    all_imgs, nfe_log = [], []
    for label, rtol, atol in TOL_CONFIGS:
        print(f"\nODE  {label}  rtol={rtol}  atol={atol} …", flush=True)
        sampler_fn = ode_sampler(sde, score_fn, shape, inverse_scaler, rtol=rtol, atol=atol, eps=eps)
        rng, k = jax.random.split(rng)
        imgs, nfe = sampler_fn(k, z=z)
        imgs = np.clip(np.array(imgs).reshape(N_IMAGES, H, H, C), 0.0, 1.0).astype(np.float32)
        all_imgs.append(imgs)
        nfe_log.append((label, nfe))
        print(f"  NFE={nfe}  done.", flush=True)

    all_imgs = np.stack(all_imgs, axis=0)
    np.save(str(out / 'nfe_sweep.npy'), all_imgs)

    log_lines = [f"{lbl}  NFE={nfe}" for lbl, nfe in nfe_log]
    (out / 'nfe_log.txt').write_text('\n'.join(log_lines) + '\n')
    print('\n'.join(log_lines))

    n_tol = len(TOL_CONFIGS)
    fig, axes = plt.subplots(N_IMAGES, n_tol, figsize=(n_tol * 2, N_IMAGES * 2))
    for col, (label, _, _) in enumerate(TOL_CONFIGS):
        axes[0, col].set_title(f"{label}\nNFE={nfe_log[col][1]}", fontsize=7)
        for row in range(N_IMAGES):
            img = all_imgs[col, row]
            axes[row, col].imshow(img if C == 3 else img[..., 0], cmap=None if C == 3 else 'gray')
            axes[row, col].axis('off')

    plt.tight_layout()
    grid_path = out / f'nfe_sweep_grid_{args.seed}.png'
    plt.savefig(str(grid_path), dpi=300, bbox_inches='tight')
    print(f"Saved {grid_path}")


if __name__ == '__main__':
    main()
