import argparse, pathlib
import numpy as np
import jax, jax.numpy as jnp
import orbax.checkpoint as ocp
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import get_config
from sde import get_sde
from utils import batch_mul
from model import UNet
from score import get_score_fn
from datasets import get_data_scaler, get_data_inverse_scaler, get_dataset
from samplers import (
    EulerMaruyamaPredictor, ReverseDiffusionPredictor, AncestralSamplingPredictor,
    LangevinCorrector, Corrector, Predictor, pc_sampler, ode_sampler,
)

# Best PC configs from score_sde paper Table 1 / appendix
BEST_PC = {
    ('vesde',    True):  ('reverse_diffusion',  'langevin', 0.16),
    ('vesde',    False): ('reverse_diffusion',  'langevin', 0.16),
    ('vpsde',    True):  ('euler_maruyama',     'none',     0.16),
    ('vpsde',    False): ('ancestral_sampling', 'none',     0.16),
    ('subvpsde', True):  ('euler_maruyama',     'none',     0.16),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_ckpt(ckpt_path, config):
    H, C = config.data.image_size, config.data.num_channels
    model = UNet(config=config)
    dummy = model.init(jax.random.PRNGKey(0), jnp.ones((1, H, H, C)), jnp.ones(1))
    step = int(pathlib.Path(ckpt_path).name)
    default_path = str(pathlib.Path(ckpt_path).resolve() / 'default')
    checkpointer = ocp.PyTreeCheckpointer()
    restored = checkpointer.restore(default_path)
    ema_params = jax.device_put(restored['ema_params'], jax.devices()[0])
    return model, ema_params, step


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


def save_grid(imgs, path, nrow=8):
    """Save float32 [0,1] (N,H,W,C) images as an nrow-column PNG grid."""
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
    parser.add_argument('--ckpt',    required=True,
                        help='Checkpoint step path, e.g. runs/ddpm_cifar10_low/ckpt/200000')
    parser.add_argument('--config',  required=True,
                        help='Config name from config.py, e.g. vpsde_ddpm_disc')
    parser.add_argument('--snap_steps', type=int, nargs='+',
                        default=[14, 86, 100, 300, 500, 548, 700, 900],
                        help='PC step indices to record trajectories at (anchor steps)')
    parser.add_argument('--seed',    type=int, default=42)
    parser.add_argument('--out_dir', default=None,
                        help='Output directory (default: assets/samples/<config>-<step>)')
    args = parser.parse_args()

    config = get_config(args.config)
    H, C = config.data.image_size, config.data.num_channels
    sde, eps = get_sde(config)
    inverse_scaler = get_data_inverse_scaler(config.data.centered)

    pred_type, corr_type, snr = BEST_PC.get(
        (config.training.sde, config.training.continuous),
        ('euler_maruyama', 'none', 0.16)
    )
    if config.data.dataset in ['celeba', 'celeb_a']:
        # CelebA uses 0.17 (vs 0.16 for CIFAR-10) per score_sde/configs/ve/celeba_ncsnpp.py
        snr = 0.17
    print(f"Best PC: predictor={pred_type}  corrector={corr_type}  snr={snr}")

    print(f"Loading {args.ckpt} …")
    model, ema_params, step = load_ckpt(args.ckpt, config)
    score_fn = get_score_fn(sde, model, ema_params,
                            train=False, continuous=config.training.continuous)

    out_dir = pathlib.Path(args.out_dir) if args.out_dir else pathlib.Path(f'assets/samples/{args.config}-{step}')
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = jax.random.PRNGKey(args.seed)

    # ── 1. 8×8 grid ───────────────────────────────────────────────────────────
    print("Generating 8×8 grid (64 samples) …")
    shape64 = (64, H, H, C)
    sampler = pc_sampler(sde, shape64,
                         build_predictor(pred_type, sde, score_fn),
                         build_corrector(corr_type, sde, score_fn, snr),
                         inverse_scaler, denoise=True, eps=eps)
    rng, k = jax.random.split(rng)
    imgs, _ = sampler(k)
    save_grid(np.array(imgs), out_dir / 'grid.png', nrow=8)

    # ── 2. Trajectory: 4 samples × n_snaps groups of 3 ───────────────────────
    # Two consecutive CPCP steps per group: C(s)→P(s)→C(s+1)→P(s+1)
    # Total images: 4 rows × (n_snaps*4) cols
    cap_anchors = sorted(set(args.snap_steps))
    n_snaps = len(cap_anchors)
    print(f"Generating trajectory (4 samples, {n_snaps} snapshot groups @ steps {cap_anchors}) …")
    n_traj = 4
    shape_t = (n_traj, H, H, C)

    _pred = jax.jit(build_predictor(pred_type, sde, score_fn).update_fn)
    _corr = jax.jit(build_corrector(corr_type, sde, score_fn, snr).update_fn)

    cap_set = set()
    for s in cap_anchors:
        cap_set.add(s)
        if s + 1 < sde.N:
            cap_set.add(s + 1)

    rng, k = jax.random.split(rng)
    x = sde.prior_sampling(k, shape_t)
    ts = np.linspace(sde.T, eps, sde.N)

    snaps_pred = {}   # step_idx -> (n_traj, H, W, C) float32 [0,1]
    snaps_corr = {}

    for i, t_val in enumerate(ts):
        vec_t = jnp.full((n_traj,), t_val)
        rng, k = jax.random.split(rng)
        x_c, _ = _corr(k, x, vec_t)
        rng, k = jax.random.split(rng)
        x_p, _ = _pred(k, x_c, vec_t)

        if i in cap_set:
            snaps_corr[i] = np.clip(np.array(inverse_scaler(x_c)), 0, 1)
            snaps_pred[i] = np.clip(np.array(inverse_scaler(x_p)), 0, 1)

        x = x_p

    # Assemble grid: n_traj rows × (n_snaps*4) cols
    # group j: C(s)  P(s)  C(s+1)  P(s+1)
    ncols = n_snaps * 4
    canvas = np.ones((n_traj * H, ncols * H, C), dtype=np.float32)
    for j, s in enumerate(cap_anchors):
        for row in range(n_traj):
            r0, r1 = row * H, (row + 1) * H
            c = j * 4 * H
            canvas[r0:r1, c       : c +   H] = snaps_corr[s    ][row]  # C(s)
            canvas[r0:r1, c +   H : c + 2*H] = snaps_pred[s    ][row]  # P(s)
            if s + 1 in snaps_corr:
                canvas[r0:r1, c + 2*H : c + 3*H] = snaps_corr[s + 1][row]  # C(s+1)
                canvas[r0:r1, c + 3*H : c + 4*H] = snaps_pred[s + 1][row]  # P(s+1)

    traj_path = out_dir / 'trajectory.png'
    plt.imsave(str(traj_path), canvas)
    print(f"  Saved {traj_path}")
    print(f"  Layout: {n_traj} rows × {ncols} cols")
    print(f"  Groups (step pairs): {[(s, s+1) for s in cap_anchors]}")


    # ── 3. Perturb 20 test images → latent, recover via ODE ──────────────────
    print("Perturbing and recovering 20 test images via probability flow ODE …")
    N_REAL = 20

    # Load N_REAL images from eval split
    _, eval_ds = get_dataset(config)
    scaler = get_data_scaler(config.data.centered)
    real_imgs = []
    for batch in eval_ds:
        imgs = scaler(batch).numpy()          # (B, H, H, C) in model input space
        real_imgs.append(imgs)
        if sum(len(b) for b in real_imgs) >= N_REAL:
            break
    real_imgs = np.concatenate(real_imgs, axis=0)[:N_REAL]   # (20, H, H, C)

    # Perturb to t=T using marginal_prob: x_T = a_T*x_0 + sigma_T*noise
    rng, k = jax.random.split(rng)
    x0 = jnp.array(real_imgs)
    vec_T = jnp.full((N_REAL,), sde.T)
    a_T, sig_T = sde.marginal_prob(x0, vec_T)
    noise = jax.random.normal(k, x0.shape)
    x_T = a_T + batch_mul(sig_T, noise)              # latent (noisy image at t=T)

    # Recover via probability flow ODE starting from x_T
    ode_fn = ode_sampler(sde, score_fn, (N_REAL, H, H, C), inverse_scaler, eps=eps)
    rng, k = jax.random.split(rng)
    recovered, nfe = ode_fn(k, z=x_T)
    recovered = np.array(recovered)
    print(f"  ODE recovery NFE: {nfe}")

    # Save three-row grid: originals | latents | recovered
    orig_disp    = np.clip(np.array(inverse_scaler(x0)), 0, 1)
    latent_disp  = np.clip(np.array(inverse_scaler(x_T)), 0, 1)

    def _row(imgs_arr, n=N_REAL):
        row = np.ones((H, n * H, C), dtype=np.float32)
        for i, img in enumerate(imgs_arr[:n]):
            row[:, i*H:(i+1)*H] = np.clip(img, 0, 1)
        return row

    canvas = np.concatenate([
        _row(orig_disp),
        _row(latent_disp),
        _row(recovered),
    ], axis=0)   # (3H, 20H, C)

    pr_path = out_dir / 'perturb_recover.png'
    plt.imsave(str(pr_path), canvas)
    print(f"  Saved {pr_path}  (rows: original | latent t=T | ODE recovered)")

    # Also save latents as numpy for inspection
    np.save(str(out_dir / 'latents.npy'), np.array(x_T))
    print(f"  Saved {out_dir}/latents.npy  shape={x_T.shape}")


if __name__ == '__main__':
    main()
