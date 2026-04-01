"""
python exp/inpaint.py --ckpt runs/ddpm_cifar10_low/ckpt/200000 \
                      --config vpsde_ddpm_disc \
                      --mask right_half
"""
import argparse, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import numpy as np
import jax, jax.numpy as jnp
import orbax.checkpoint as ocp
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from config   import get_config
from sde      import get_sde
from model    import UNet
from score    import get_score_fn
from utils    import batch_mul
from datasets import get_data_scaler, get_data_inverse_scaler, get_dataset
from samplers import (
    EulerMaruyamaPredictor, ReverseDiffusionPredictor, AncestralSamplingPredictor,
    LangevinCorrector, Corrector, Predictor,
)

BEST_PC = {
    ('vesde',    True):  ('reverse_diffusion',  'langevin', 0.16),
    ('vesde',    False): ('reverse_diffusion',  'langevin', 0.16),
    ('vpsde',    True):  ('euler_maruyama',     'none',     0.16),
    ('vpsde',    False): ('ancestral_sampling', 'none',     0.16),
    ('subvpsde', True):  ('euler_maruyama',     'none',     0.16),
}

MASK_TYPES = ('right_half', 'bottom_half', 'center')
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')


def load_ckpt(ckpt_path, config):
    model      = UNet(config=config)
    restored   = ocp.StandardCheckpointer().restore(str(pathlib.Path(ckpt_path).resolve() / 'default'))
    ema_params = jax.device_put(restored['ema_params'], jax.devices()[0])
    return model, ema_params


def build_predictor(pred_type, sde, score_fn):
    if pred_type == 'ancestral_sampling':
        return AncestralSamplingPredictor(sde, score_fn)
    if pred_type == 'reverse_diffusion':
        return ReverseDiffusionPredictor(sde, score_fn, False)
    if pred_type == 'euler_maruyama':
        return EulerMaruyamaPredictor(sde, score_fn, False)
    return Predictor()


def build_corrector(corr_type, sde, score_fn, snr):
    if corr_type == 'langevin':
        return LangevinCorrector(sde, score_fn, snr, n_steps=1)
    return Corrector()


def make_mask(mask_type, H, W, C, N):
    m = np.ones((H, W, C), dtype=np.float32)
    if mask_type == 'right_half':
        m[:, W // 2:, :] = 0.0
    elif mask_type == 'bottom_half':
        m[H // 2:, :, :] = 0.0
    elif mask_type == 'center':
        h0, h1 = H // 4, 3 * H // 4
        w0, w1 = W // 4, 3 * W // 4
        m[h0:h1, w0:w1, :] = 0.0
    return jnp.array(np.broadcast_to(m, (N, H, W, C)))


def load_images_from_dir(image_dir, H, C):
    d = pathlib.Path(image_dir)
    img_paths = sorted(p for p in d.iterdir()
                       if p.suffix.lower() in IMAGE_EXTENSIONS and '-mask' not in p.stem)

    images, masks, names = [], [], []
    for ip in img_paths:
        mask_stem = ip.stem + '-mask'
        mp = None
        for ext in [ip.suffix] + [e for e in IMAGE_EXTENSIONS if e != ip.suffix]:
            candidate = d / (mask_stem + ext)
            if candidate.exists():
                mp = candidate
                break
        if mp is None:
            print(f"  Warning: no mask found for {ip.name}, skipping.")
            continue

        def _load_resize(path, mode):
            img = Image.open(path).convert(mode)
            w, h = img.size
            s = min(w, h)
            left, top = (w - s) // 2, (h - s) // 2
            img = img.crop((left, top, left + s, top + s))
            img = img.resize((H, H), Image.BICUBIC)
            return np.array(img, dtype=np.float32) / 255.0

        mode     = 'RGB' if C == 3 else 'L'
        img_arr  = _load_resize(ip, mode)
        mask_arr = _load_resize(mp, 'RGB')

        if img_arr.ndim == 2:
            img_arr = img_arr[:, :, None]

        is_green    = (mask_arr[..., 1] > 0.5) & (mask_arr[..., 0] < 0.5) & (mask_arr[..., 2] < 0.5)
        binary_mask = np.broadcast_to((~is_green).astype(np.float32)[:, :, None], (H, H, C)).copy()

        images.append(img_arr)
        masks.append(binary_mask)
        names.append(ip.stem)

    return np.stack(images, 0), np.stack(masks, 0), names


def save_grid(imgs, path, nrow):
    n, H, W, C = imgs.shape
    nrows  = int(np.ceil(n / nrow))
    canvas = np.ones((nrows * H, nrow * W, C), dtype=np.float32)
    for idx, img in enumerate(imgs):
        r, c = divmod(idx, nrow)
        canvas[r*H:(r+1)*H, c*W:(c+1)*W] = np.clip(img, 0, 1)
    plt.imsave(str(path), canvas)
    print(f"  Saved {path}")


def pc_inpaint(rng, data, mask, sde, predictor, corrector, inverse_scaler, denoise=True, eps=1e-3):
    N = data.shape[0]
    timesteps = jnp.linspace(sde.T, eps, sde.N)

    @jax.jit
    def pred_update(rng, x, t):
        vec_t = jnp.full((N,), t)
        rng, k = jax.random.split(rng)
        x, x_mean = predictor.update_fn(k, x, vec_t)
        mean_data, std = sde.marginal_prob(data, vec_t)
        rng, k = jax.random.split(rng)
        noisy_data = mean_data + batch_mul(jax.random.normal(k, data.shape), std)
        x      = x      * (1.0 - mask) + noisy_data * mask
        x_mean = x_mean * (1.0 - mask) + mean_data  * mask
        return rng, x, x_mean

    @jax.jit
    def corr_update(rng, x, t):
        vec_t = jnp.full((N,), t)
        rng, k = jax.random.split(rng)
        x, x_mean = corrector.update_fn(k, x, vec_t)
        mean_data, std = sde.marginal_prob(data, vec_t)
        rng, k = jax.random.split(rng)
        noisy_data = mean_data + batch_mul(jax.random.normal(k, data.shape), std)
        x      = x      * (1.0 - mask) + noisy_data * mask
        x_mean = x_mean * (1.0 - mask) + mean_data  * mask
        return rng, x, x_mean

    rng, k = jax.random.split(rng)
    x_init = data * mask + sde.prior_sampling(k, data.shape) * (1.0 - mask)
    x, x_mean = x_init, x_init
    for i, t in enumerate(timesteps):
        rng, x, x_mean = corr_update(rng, x, t)
        rng, x, x_mean = pred_update(rng, x, t)
        if (i + 1) % 100 == 0:
            print(f"    step {i+1}/{sde.N}")

    return jnp.clip(inverse_scaler(x_mean if denoise else x), 0.0, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt',       required=True)
    parser.add_argument('--config',     required=True)
    parser.add_argument('--mask',       default='right_half', choices=MASK_TYPES)
    parser.add_argument('--n',          type=int, default=16)
    parser.add_argument('--image_dir',  default=None)
    parser.add_argument('--seed',       type=int, default=0)
    args = parser.parse_args()

    config = get_config(args.config)
    H, C   = config.data.image_size, config.data.num_channels
    sde, eps = get_sde(config)
    scaler         = get_data_scaler(config.data.centered)
    inverse_scaler = get_data_inverse_scaler(config.data.centered)

    pred_type, corr_type, snr = BEST_PC.get(
        (config.training.sde, config.training.continuous), ('euler_maruyama', 'none', 0.16)
    )
    if config.data.dataset.lower() in ('celeba', 'celeb_a'):
        snr = 0.17
    print(f"Predictor={pred_type}  corrector={corr_type}  snr={snr}")

    print(f"Loading {args.ckpt} …")
    model, ema_params = load_ckpt(args.ckpt, config)
    score_fn = get_score_fn(sde, model, ema_params, train=False, continuous=config.training.continuous)

    pred = build_predictor(pred_type, sde, score_fn)
    corr = build_corrector(corr_type, sde, score_fn, snr)

    if args.image_dir is not None:
        print(f"Loading images from {args.image_dir} …")
        raw_imgs, mask, img_names = load_images_from_dir(args.image_dir, H, C)
        mask   = jnp.array(mask)
        n_imgs = len(raw_imgs)
        print(f"  Loaded {n_imgs} image(s): {img_names}")
    else:
        print(f"Loading {args.n} test images …")
        _, eval_ds = get_dataset(config)
        raw_imgs = []
        for batch in eval_ds:
            raw_imgs.append(np.array(batch))
            if sum(len(b) for b in raw_imgs) >= args.n:
                break
        raw_imgs   = np.concatenate(raw_imgs, 0)[:args.n]
        n_imgs     = args.n
        img_names  = None
        mask       = make_mask(args.mask, H, H, C, n_imgs)

    data_scaled = jnp.array(scaler(raw_imgs))

    print(f"Inpainting {n_imgs} image(s) …")
    rng       = jax.random.PRNGKey(args.seed)
    inpainted = np.array(pc_inpaint(rng, data_scaled, mask, sde, pred, corr, inverse_scaler, eps=eps))

    step    = pathlib.Path(args.ckpt).name
    out_dir = pathlib.Path(f'assets/samples/{args.config}-{step}/inpainting')
    out_dir.mkdir(parents=True, exist_ok=True)

    orig   = np.clip(raw_imgs, 0, 1)
    masked = np.clip(raw_imgs * np.array(mask), 0, 1)

    if args.image_dir and img_names:
        for i, name in enumerate(img_names):
            strip_i = np.concatenate([orig[i], masked[i], inpainted[i]], axis=1)
            plt.imsave(str(out_dir / f'{name}_strip.png'), np.clip(strip_i, 0, 1))
            plt.imsave(str(out_dir / f'{name}_inpainted.png'), np.clip(inpainted[i], 0, 1))
            print(f"  Saved {name}_strip.png")
    else:
        nrow = min(n_imgs, 8)
        save_grid(orig,      out_dir / 'original.png',  nrow=nrow)
        save_grid(masked,    out_dir / 'masked.png',    nrow=nrow)
        save_grid(inpainted, out_dir / 'inpainted.png', nrow=nrow)

        def to_row(imgs):
            n, H2, W2, C2 = imgs.shape
            row = np.ones((H2, n * W2, C2), dtype=np.float32)
            for i, img in enumerate(imgs):
                row[:, i*W2:(i+1)*W2] = np.clip(img, 0, 1)
            return row

        plt.imsave(str(out_dir / 'strip.png'),
                   np.concatenate([to_row(orig), to_row(masked), to_row(inpainted)], axis=0))
        print(f"  Saved strip.png")
    print("Done.")


if __name__ == '__main__':
    main()
