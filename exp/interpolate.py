import argparse, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import numpy as np
from PIL import Image
from scipy import integrate
import jax, jax.numpy as jnp
import orbax.checkpoint as ocp
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import get_config
from sde import get_sde
from model import UNet
from score import get_score_fn
from datasets import get_data_scaler, get_data_inverse_scaler, get_dataset

VARIANTS = {
    'disc': ('vesde_ncsnpp_celeba_disc', 'runs/ncsnpp_celeba_disc/ckpt/50000'),
    'cont': ('vesde_ncsnpp_celeba_cont', 'runs/ncsnpp_celeba_cont/ckpt/50000'),
}
IMG_DIR = pathlib.Path('interpolate')


def load_ckpt(ckpt_path, config):
    model      = UNet(config=config)
    restored   = ocp.PyTreeCheckpointer().restore(str(pathlib.Path(ckpt_path).resolve() / 'default'))
    ema_params = jax.device_put(restored['ema_params'], jax.devices()[0])
    return model, ema_params


def load_image(path, image_size):
    img = Image.open(path).convert('RGB')
    w, h = img.size
    if min(w, h) >= 140:
        crop = 140
        left, top = (w - crop) // 2, (h - crop) // 2
        img = img.crop((left, top, left + crop, top + crop))
    img = img.resize((image_size, image_size), Image.BILINEAR)
    return np.array(img, dtype=np.float32) / 255.0


def slerp(z1, z2, alpha):
    z1_flat = z1.ravel()
    z2_flat = z2.ravel()
    omega = jnp.arccos(jnp.clip(
        jnp.dot(z1_flat, z2_flat) / (jnp.linalg.norm(z1_flat) * jnp.linalg.norm(z2_flat) + 1e-8),
        -1.0, 1.0,
    ))
    sin_omega = jnp.sin(omega) + 1e-8
    coeff1 = jnp.sin((1.0 - alpha) * omega) / sin_omega
    coeff2 = jnp.sin(alpha * omega) / sin_omega
    return (coeff1 * z1_flat + coeff2 * z2_flat).reshape(z1.shape)


def tweedie_denoise(z1, z2, score_fn, sde, t_enc):
    vec_t = jnp.full((1,), t_enc)
    _, sigma_t = sde.marginal_prob(z1, t_enc)
    x0_1 = z1 + sigma_t ** 2 * score_fn(z1, vec_t)
    x0_2 = z2 + sigma_t ** 2 * score_fn(z2, vec_t)
    eps1  = (z1 - x0_1) / sigma_t
    eps2  = (z2 - x0_2) / sigma_t
    return x0_1, x0_2, eps1, eps2, sigma_t


def tweedie_blend(x0_1, x0_2, eps1, eps2, sigma_t, alpha):
    x0_alpha  = (1.0 - alpha) * x0_1 + alpha * x0_2
    eps_alpha = slerp(eps1[0], eps2[0], alpha)[None]
    return x0_alpha + sigma_t * eps_alpha


def save_grid(imgs, path, nrow):
    n, H, W, C = imgs.shape
    nrows  = int(np.ceil(n / nrow))
    canvas = np.ones((nrows * H, nrow * W, C), dtype=np.float32)
    for idx, img in enumerate(imgs):
        r, c = divmod(idx, nrow)
        canvas[r*H:(r+1)*H, c*W:(c+1)*W] = np.clip(img, 0, 1)
    plt.imsave(str(path), canvas)
    print(f"  Saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_interp',      type=int,   default=8)
    parser.add_argument('--out_dir',       default=None)
    parser.add_argument('--rtol',          type=float, default=1e-5)
    parser.add_argument('--atol',          type=float, default=1e-5)
    parser.add_argument('--n_celeba',      type=int,   default=None)
    parser.add_argument('--seed',          type=int,   default=0)
    parser.add_argument('--variant',       choices=['disc', 'cont'], default='disc')
    parser.add_argument('--interp_method', choices=['slerp', 'tweedie'], default='slerp')
    parser.add_argument('--t_enc',         type=float, default=None)
    parser.add_argument('--no_denoise',    action='store_true', default=False)
    parser.add_argument('--show_originals', action='store_true', default=False)
    args = parser.parse_args()

    cfg_name, ckpt_path = VARIANTS[args.variant]
    out_dir = pathlib.Path(args.out_dir) if args.out_dir else pathlib.Path(f'assets/eval/ncsnpp_celeba_{args.variant}/interpolation')
    out_dir.mkdir(parents=True, exist_ok=True)

    config = get_config(cfg_name)
    H, C   = config.data.image_size, config.data.num_channels
    shape  = (1, H, H, C)
    sde, eps = get_sde(config)
    scaler         = get_data_scaler(config.data.centered)
    inverse_scaler = get_data_inverse_scaler(config.data.centered)

    t_enc = args.t_enc if args.t_enc is not None else sde.T
    print(f"Encoding time t_enc={t_enc:.4f}  (sde.T={sde.T})")

    print(f"Loading {ckpt_path} …")
    model, ema_params = load_ckpt(ckpt_path, config)
    score_fn = get_score_fn(sde, model, ema_params, train=False, continuous=config.training.continuous)

    @jax.jit
    def drift_fn(x, t):
        f, _ = sde.reverse_sde(x, t, score_fn(x, t), probability_flow=True)
        return f

    def ode_fn(t, x_flat):
        return np.array(drift_fn(jnp.asarray(x_flat).reshape(shape), jnp.full((1,), t))).reshape(-1)

    def encode(x):
        sol = integrate.solve_ivp(ode_fn, (eps, t_enc), np.array(x).reshape(-1), method='RK45', rtol=args.rtol, atol=args.atol)
        print(f"    encode NFE: {sol.nfev}")
        return jnp.asarray(sol.y[:, -1]).reshape(shape)

    def decode(z, denoise=True):
        sol = integrate.solve_ivp(ode_fn, (t_enc, eps), np.array(z).reshape(-1), method='RK45', rtol=args.rtol, atol=args.atol)
        print(f"    decode NFE: {sol.nfev}")
        x_eps = jnp.asarray(sol.y[:, -1]).reshape(shape)
        if denoise:
            _, sigma_eps = sde.marginal_prob(x_eps, eps)
            x_eps = x_eps + sigma_eps ** 2 * score_fn(x_eps, jnp.full((1,), eps))
        return inverse_scaler(x_eps)

    if args.n_celeba:
        print(f"Loading {args.n_celeba} random pairs from CelebA eval set …")
        _, eval_ds = get_dataset(config)
        imgs_np = np.stack(list(
            eval_ds.unbatch().shuffle(10_000, seed=args.seed).take(args.n_celeba * 2).as_numpy_iterator()
        ))
        pairs = [
            (jnp.asarray(scaler(imgs_np[2*i]))[None], jnp.asarray(scaler(imgs_np[2*i+1]))[None], f'celeba_{i+1:02d}')
            for i in range(args.n_celeba)
        ]
    else:
        indices = sorted({p.stem.split('-')[0] for p in IMG_DIR.glob('*-1.jpg')}, key=lambda s: int(s))
        pairs = []
        for idx in indices:
            p1, p2 = IMG_DIR / f'{idx}-1.jpg', IMG_DIR / f'{idx}-2.jpg'
            if not p2.exists():
                print(f"  Warning: {p1.name} has no matching {p2.name}, skipping.")
                continue
            pairs.append((
                jnp.asarray(scaler(load_image(p1, H)))[None],
                jnp.asarray(scaler(load_image(p2, H)))[None],
                f'{int(idx):02d}',
            ))

    all_rows = []
    for pair_idx, (x1, x2, label) in enumerate(pairs, 1):
        print(f"\nPair {pair_idx}/{len(pairs)}: {label}")
        print("  Encoding …")
        z1, z2 = encode(x1), encode(x2)

        alphas = np.linspace(0.0, 1.0, args.n_interp)
        print(f"  Interpolating ({args.interp_method}) and decoding {args.n_interp} latents …")
        if args.interp_method == 'tweedie':
            tweedie_components = tweedie_denoise(z1, z2, score_fn, sde, t_enc)
        imgs_list = []
        for i, a in enumerate(alphas):
            alpha = float(a)
            z_a = slerp(z1[0], z2[0], alpha)[None] if args.interp_method == 'slerp' else tweedie_blend(*tweedie_components, alpha)
            imgs_list.append(np.clip(np.array(decode(z_a, denoise=not args.no_denoise)), 0, 1))
            print(f"    step {i+1}/{args.n_interp}")
        imgs = np.concatenate(imgs_list, axis=0)

        if args.show_originals:
            orig1 = np.clip(np.array(inverse_scaler(x1)), 0, 1)
            orig2 = np.clip(np.array(inverse_scaler(x2)), 0, 1)
            imgs  = np.concatenate([orig1, imgs, orig2], axis=0)

        nrow = args.n_interp + (2 if args.show_originals else 0)
        all_rows.append(imgs)
        save_grid(imgs, out_dir / f'pair_{label}.png', nrow=nrow)

    if not all_rows:
        print(f"No pairs found. Pass --n_celeba N or add images to {IMG_DIR}/")
        return

    nrow     = args.n_interp + (2 if args.show_originals else 0)
    all_imgs = np.concatenate(all_rows, axis=0)
    save_grid(all_imgs, out_dir / 'interpolation_grid.png', nrow=nrow)
    print(f"\nDone. {len(pairs)} pairs × {nrow} cols → {out_dir}/interpolation_grid.png")


if __name__ == '__main__':
    main()
