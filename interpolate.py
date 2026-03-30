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
        """Real image → latent: integrate probability-flow ODE forward (ε → T)."""
        sol = integrate.solve_ivp(
            ode_fn, (eps, sde.T), np.array(x).reshape(-1),
            method='RK45', rtol=args.rtol, atol=args.atol
        )
        print(f"    encode NFE: {sol.nfev}")
        return jnp.asarray(sol.y[:, -1]).reshape(shape)

    def decode(z):
        """Latent → image: integrate probability-flow ODE backward (T → ε)."""
        sol = integrate.solve_ivp(
            ode_fn, (sde.T, eps), np.array(z).reshape(-1),
            method='RK45', rtol=args.rtol, atol=args.atol
        )
        print(f"    decode NFE: {sol.nfev}")
        x = jnp.asarray(sol.y[:, -1]).reshape(shape)
        return inverse_scaler(x)

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
        pairs = []
        idx = 1
        while True:
            p1 = IMG_DIR / f'{idx}-1.jpg'
            p2 = IMG_DIR / f'{idx}-2.jpg'
            if not p1.exists() or not p2.exists():
                break
            pairs.append((
                jnp.asarray(scaler(load_image(p1, H)))[None],
                jnp.asarray(scaler(load_image(p2, H)))[None],
                f'{idx:02d}',
            ))
            idx += 1

    # ── Interpolation loop ────────────────────────────────────────────────────
    all_rows = []

    for pair_idx, (x1, x2, label) in enumerate(pairs, 1):
        print(f"\nPair {pair_idx}/{len(pairs)}: {label}")

        # Encode both images into the latent space
        print("  Encoding …")
        z1 = encode(x1)  # (1, H, H, C) at t=T
        z2 = encode(x2)

        # Slerp between the two latent codes
        alphas   = np.linspace(0.0, 1.0, args.n_interp)
        z_interp = jnp.stack(
            [slerp(z1[0], z2[0], float(a)) for a in alphas], axis=0
        )  # (n_interp, H, H, C)

        # Decode each interpolated latent back to image space
        print(f"  Decoding {args.n_interp} latents …")
        imgs_list = []
        for i in range(args.n_interp):
            img = decode(z_interp[i:i+1])
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
