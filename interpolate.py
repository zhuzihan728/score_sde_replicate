import argparse, pathlib
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

# ── Config ────────────────────────────────────────────────────────────────────
VARIANTS = {
    'disc': ('vesde_ncsnpp_celeba_disc', 'runs/ncsnpp_celeba_disc/ckpt/50000'),
    'cont': ('vesde_ncsnpp_celeba_cont', 'runs/ncsnpp_celeba_cont/ckpt/50000'),
}
IMG_DIR = pathlib.Path('interpolate')

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_ckpt(ckpt_path, config):
    model = UNet(config=config)
    default_path = str(pathlib.Path(ckpt_path).resolve() / 'default')
    checkpointer = ocp.PyTreeCheckpointer()
    restored = checkpointer.restore(default_path)
    ema_params = jax.device_put(restored['ema_params'], jax.devices()[0])
    return model, ema_params


def load_image(path, image_size):
    """Load image with CelebA-style preprocessing: center-crop 140, resize to image_size."""
    img = Image.open(path).convert('RGB')
    w, h = img.size
    if min(w, h) >= 140:
        crop = 140
        left = (w - crop) // 2
        top  = (h - crop) // 2
        img  = img.crop((left, top, left + crop, top + crop))
    img = img.resize((image_size, image_size), Image.BILINEAR)
    return np.array(img, dtype=np.float32) / 255.0  # (H, W, C) in [0, 1]


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


def tweedie_denoise(z1, z2, score_fn, sde, t_enc):
    """Compute Tweedie x̂_0 estimates and noise residuals for both endpoints.

    For VE SDE at t_enc: p_{t_enc}(z | x_0) = N(x_0, σ(t_enc)² I), so the
    Tweedie posterior mean estimate is  x̂_0 = z + σ(t_enc)² · s_θ(z, t_enc).
    Returns components that are independent of alpha — call once per pair.
    """
    vec_t = jnp.full((1,), t_enc)
    _, sigma_t = sde.marginal_prob(z1, t_enc)  # scalar σ(t_enc)

    x0_1 = z1 + sigma_t ** 2 * score_fn(z1, vec_t)
    x0_2 = z2 + sigma_t ** 2 * score_fn(z2, vec_t)
    eps1  = (z1 - x0_1) / sigma_t
    eps2  = (z2 - x0_2) / sigma_t

    return x0_1, x0_2, eps1, eps2, sigma_t


def tweedie_blend(x0_1, x0_2, eps1, eps2, sigma_t, alpha):
    """Blend pre-computed Tweedie components at interpolation weight alpha.

    Steps:
      1. Linear-interpolate the image components.
      2. Slerp the noise residuals independently.
      3. Recombine: z_α = x̂_0^(α) + σ(t_enc) · ε_α.
    """
    x0_alpha  = (1.0 - alpha) * x0_1 + alpha * x0_2
    eps_alpha = slerp(eps1[0], eps2[0], alpha)[None]
    return x0_alpha + sigma_t * eps_alpha


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
    parser.add_argument('--n_interp', type=int, default=8,
                        help='Interpolation steps per pair (including endpoints)')
    parser.add_argument('--out_dir',  default=None)
    parser.add_argument('--rtol',     type=float, default=1e-5)
    parser.add_argument('--atol',     type=float, default=1e-5)
    parser.add_argument('--n_celeba', type=int, default=None,
                        help='Use N random pairs from CelebA eval set instead of interpolate/')
    parser.add_argument('--seed',     type=int, default=0,
                        help='Shuffle seed for --n_celeba')
    parser.add_argument('--variant', choices=['disc', 'cont'], default='disc')
    parser.add_argument('--interp_method', choices=['slerp', 'tweedie'], default='slerp',
                        help='Interpolation method: slerp (spherical) or tweedie (denoising+re-noising)')
    parser.add_argument('--t_enc', type=float, default=None,
                        help='Encoding time in [0, T]; defaults to sde.T (full noise). '
                             'Use e.g. 0.5 to encode only halfway.')
    parser.add_argument('--no_denoise', action='store_true', default=False,
                        help='Skip final Tweedie denoising step at t=ε after decoding')
    parser.add_argument('--show_originals', action='store_true', default=False,
                        help='Prepend/append original images to each row for reference')
    args = parser.parse_args()

    cfg_name, ckpt_path = VARIANTS[args.variant]
    out_dir = pathlib.Path(args.out_dir) if args.out_dir else pathlib.Path(f'assets/eval/ncsnpp_celeba_{args.variant}/interpolation')
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
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
    score_fn = get_score_fn(sde, model, ema_params,
                            train=False, continuous=config.training.continuous)

    # ── Probability-flow ODE (shared by encoder and decoder) ──────────────────
    # drift_fn returns f(x,t) - 0.5·g(t)²·score(x,t)  (same for both directions)
    @jax.jit
    def drift_fn(x, t):
        f, _ = sde.reverse_sde(x, t, score_fn(x, t), probability_flow=True)
        return f

    def ode_fn(t, x_flat):
        x_r  = jnp.asarray(x_flat).reshape(shape)
        vec_t = jnp.full((1,), t)
        return np.array(drift_fn(x_r, vec_t)).reshape(-1)

    def encode(x):
        """Real image → latent: integrate probability-flow ODE forward (ε → t_enc)."""
        sol = integrate.solve_ivp(
            ode_fn, (eps, t_enc), np.array(x).reshape(-1),
            method='RK45', rtol=args.rtol, atol=args.atol
        )
        print(f"    encode NFE: {sol.nfev}")
        return jnp.asarray(sol.y[:, -1]).reshape(shape)

    def decode(z, denoise=True):
        """Latent → image: integrate probability-flow ODE backward (t_enc → ε).

        If denoise=True, applies a final Tweedie denoising step at t=ε to
        remove residual low-level noise before inverse-scaling.
        """
        sol = integrate.solve_ivp(
            ode_fn, (t_enc, eps), np.array(z).reshape(-1),
            method='RK45', rtol=args.rtol, atol=args.atol
        )
        print(f"    decode NFE: {sol.nfev}")
        x_eps = jnp.asarray(sol.y[:, -1]).reshape(shape)

        if denoise:
            vec_eps = jnp.full((1,), eps)
            _, sigma_eps = sde.marginal_prob(x_eps, eps)
            x_eps = x_eps + sigma_eps ** 2 * score_fn(x_eps, vec_eps)

        return inverse_scaler(x_eps)

    # ── Build pairs list ─────────────────────────────────────────────────────
    if args.n_celeba:
        print(f"Loading {args.n_celeba} random pairs from CelebA eval set …")
        _, eval_ds = get_dataset(config)
        imgs_np = np.stack(list(
            eval_ds.unbatch()
                   .shuffle(10_000, seed=args.seed)
                   .take(args.n_celeba * 2)
                   .as_numpy_iterator()
        ))  # (n*2, H, H, C) in [0, 1]
        pairs = [
            (jnp.asarray(scaler(imgs_np[2*i]))[None],
             jnp.asarray(scaler(imgs_np[2*i+1]))[None],
             f'celeba_{i+1:02d}')
            for i in range(args.n_celeba)
        ]
    else:
        indices = sorted({p.stem.split('-')[0] for p in IMG_DIR.glob('*-1.jpg')},
                         key=lambda s: int(s))
        pairs = []
        for idx in indices:
            p1 = IMG_DIR / f'{idx}-1.jpg'
            p2 = IMG_DIR / f'{idx}-2.jpg'
            if not p2.exists():
                print(f"  Warning: {p1.name} has no matching {p2.name}, skipping.")
                continue
            pairs.append((
                jnp.asarray(scaler(load_image(p1, H)))[None],
                jnp.asarray(scaler(load_image(p2, H)))[None],
                f'{int(idx):02d}',
            ))

    # ── Interpolation loop ────────────────────────────────────────────────────
    all_rows = []

    for pair_idx, (x1, x2, label) in enumerate(pairs, 1):
        print(f"\nPair {pair_idx}/{len(pairs)}: {label}")

        # Encode both images into the latent space
        print("  Encoding …")
        z1 = encode(x1)  # (1, H, H, C) at t=t_enc
        z2 = encode(x2)

        # Interpolate between the two latent codes and decode
        alphas = np.linspace(0.0, 1.0, args.n_interp)
        print(f"  Interpolating ({args.interp_method}) and decoding {args.n_interp} latents …")
        if args.interp_method == 'tweedie':
            tweedie_components = tweedie_denoise(z1, z2, score_fn, sde, t_enc)
        imgs_list = []
        for i, a in enumerate(alphas):
            alpha = float(a)
            if args.interp_method == 'slerp':
                z_a = slerp(z1[0], z2[0], alpha)[None]
            else:  # tweedie
                z_a = tweedie_blend(*tweedie_components, alpha)
            img = decode(z_a, denoise=not args.no_denoise)
            imgs_list.append(np.clip(np.array(img), 0, 1))
            print(f"    step {i+1}/{args.n_interp}")
        imgs = np.concatenate(imgs_list, axis=0)  # (n_interp, H, H, C)

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

    nrow = args.n_interp + (2 if args.show_originals else 0)
    all_imgs = np.concatenate(all_rows, axis=0)
    save_grid(all_imgs, out_dir / 'interpolation_grid.png', nrow=nrow)
    print(f"\nDone. {len(pairs)} pairs × {nrow} cols → {out_dir}/interpolation_grid.png")


if __name__ == '__main__':
    main()
