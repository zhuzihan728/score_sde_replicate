#!/usr/bin/env python
"""
eval.py — evaluate and visualise trained diffusion models.

Default (no flags): benchmark all models → FID + IS + 8×8 grid.

Common usage:
  python eval.py                                          # full sweep, all models
  python eval.py --model vpsde_disc                       # full eval for one model
  python eval.py --model vpsde_disc --sample-only         # 8×8 grid only, no FID/IS
  python eval.py --model vpsde_disc --plot corrector_sweep
  python eval.py --ckpt runs/my/ckpt/100000 --config vpsde_ddpm_disc --sample-only
"""

import argparse, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import numpy as np
import jax, jax.numpy as jnp
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import orbax.checkpoint as ocp
import tensorflow as tf
import tensorflow_hub as tfhub

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

# ── Constants ──────────────────────────────────────────────────────────────────
INCEPTION_TFHUB = 'https://tfhub.dev/tensorflow/tfgan/eval/inception/1'
N_SAMPLES  = 10_000
BATCH_SIZE = 1024
GRID_N     = 64
OUT_ROOT   = pathlib.Path('assets/eval')

# ── Model registry ─────────────────────────────────────────────────────────────
# (display_name, config_name, ckpt_path, celeba_only)
MODELS = [
    ('vpsde_disc',         'vpsde_ddpm_disc',         'runs/vpsde_cifar10_disc/ckpt/190000',    False),
    ('vpsde_cont',         'vpsde_ddpm_cont',          'runs/vpsde_cifar10_cont/ckpt/190000',    False),
    ('subvpsde_cont',      'subvpsde_ddpm_cont',       'runs/subvp_cifar10_cont/ckpt/190000',    False),
    ('vesde_disc',         'vesde_ddpm_disc',           'runs/vesde_cifar10_disc/ckpt/190000',    False),
    ('vesde_cont',         'vesde_ddpm_cont',           'runs/vesde_cifar10_cont/ckpt/190000',    False),
    ('ddpmpp_cont',        'vpsde_ddpmpp_cont',         'runs/ddpmpp_cifar10_cont/ckpt/160000',   False),
    ('ncsnpp_cont',        'vesde_ncsnpp_cont',         'runs/ncsnpp_cifar10_cont/ckpt/160000',   False),
    ('ncsnpp_celeba',      'vesde_ncsnpp_celeba_disc',  'runs/ncsnpp_celeba_disc/ckpt/50000',     True),
    ('ncsnpp_celeba_cont', 'vesde_ncsnpp_celeba_cont',  'runs/ncsnpp_celeba_cont/ckpt/50000',     True),
]
MODELS_BY_NAME = {m[0]: m for m in MODELS}

# Best PC per (sde, continuous)
BEST_PC = {
    ('vesde',    True):  ('reverse_diffusion',  'langevin', 0.16),
    ('vesde',    False): ('reverse_diffusion',  'langevin', 0.16),
    ('vpsde',    False): ('ancestral_sampling', 'none',     0.16),
    ('vpsde',    True):  ('euler_maruyama',     'none',     0.16),
    ('subvpsde', False): ('ancestral_sampling', 'none',     0.16),
    ('subvpsde', True):  ('euler_maruyama',     'none',     0.16),
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_ckpt(ckpt_path, config):
    H, C = config.data.image_size, config.data.num_channels
    model = UNet(config=config)
    model.init(jax.random.PRNGKey(0), jnp.ones((1, H, H, C)), jnp.ones(1))
    step = int(pathlib.Path(ckpt_path).name)
    default_path = str(pathlib.Path(ckpt_path).resolve() / 'default')
    restored = ocp.PyTreeCheckpointer().restore(default_path)
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


def build_corrector(corr_type, sde, score_fn, snr, n_steps=1):
    if corr_type == 'langevin':
        return LangevinCorrector(sde, score_fn, snr, n_steps=n_steps)
    return Corrector()


def save_grid(imgs, path, nrow=8):
    n, H, W, C = imgs.shape
    nrows = int(np.ceil(n / nrow))
    canvas = np.ones((nrows * H, nrow * W, C), dtype=np.float32)
    for idx, img in enumerate(imgs):
        r, c = divmod(idx, nrow)
        canvas[r*H:(r+1)*H, c*W:(c+1)*W] = np.clip(img, 0, 1)
    plt.imsave(str(path), canvas)
    print(f"  Saved grid → {path}")


def make_sampler(sde, shape, pred, corr, inverse_scaler, eps):
    """JIT-compiled PC sampler via fori_loop (for bulk generation / FID)."""
    n_corr = getattr(corr, 'n_steps', 0) or 0

    @jax.jit
    def _sample(rng):
        rng, k = jax.random.split(rng)
        x = sde.prior_sampling(k, shape)
        ts = jnp.linspace(sde.T, eps, sde.N)

        def step(i, val):
            rng, x, x_mean = val
            t = jnp.full((shape[0],), ts[i])
            rng, k = jax.random.split(rng)
            x, x_mean = corr.update_fn(k, x, t)
            rng, k = jax.random.split(rng)
            x, x_mean = pred.update_fn(k, x, t)
            return rng, x, x_mean

        _, x, x_mean = jax.lax.fori_loop(0, sde.N, step, (rng, x, x))
        return inverse_scaler(x_mean), sde.N * (n_corr + 1)

    return _sample


def to_uint8(imgs):
    return (np.clip(imgs, 0.0, 1.0) * 255).astype(np.uint8)


# ── Inception / FID / IS ───────────────────────────────────────────────────────

@tf.function
def _run_inception_batch(batch_uint8, model):
    x = (tf.cast(batch_uint8, tf.float32) - 127.5) / 127.5
    return model(x)


def run_inception(samples_uint8, model, batch_size=500):
    pools, logits_all = [], []
    for i in range(0, len(samples_uint8), batch_size):
        out = _run_inception_batch(tf.constant(samples_uint8[i:i+batch_size]), model)
        pools.append(out['pool_3'].numpy().reshape(len(out['pool_3']), -1))
        logits_all.append(out['logits'].numpy())
    return np.concatenate(pools, 0), np.concatenate(logits_all, 0)


def inception_score(logits):
    logits = logits.astype(np.float64)
    log_p = (logits
             - np.log(np.sum(np.exp(logits - logits.max(1, keepdims=True)), axis=1, keepdims=True))
             - logits.max(1, keepdims=True))
    p = np.exp(log_p)
    kl = (p * (log_p - np.log(p.mean(0)))).sum(1)
    return float(np.exp(kl.mean()))


def _sym_sqrtm(m, eps=1e-10):
    s, u = np.linalg.eigh(m)
    s = np.where(s < eps, 0.0, np.sqrt(np.maximum(s, 0.0)))
    return (u * s) @ u.T


def frechet_distance(act_real, act_gen):
    act_real = act_real.astype(np.float64)
    act_gen  = act_gen.astype(np.float64)
    mu_r, mu_g = act_real.mean(0), act_gen.mean(0)
    sigma_r    = np.cov(act_real, rowvar=False)
    sigma_g    = np.cov(act_gen,  rowvar=False)
    sqrt_r     = _sym_sqrtm(sigma_r)
    sqrt_trace = np.trace(_sym_sqrtm(sqrt_r @ sigma_g @ sqrt_r))
    diff       = mu_r - mu_g
    return float(diff @ diff + np.trace(sigma_r) + np.trace(sigma_g) - 2.0 * sqrt_trace)


def load_real_stats(config, inception_model):
    dataset    = config.data.dataset.lower()
    stats_path = pathlib.Path(f'assets/stats/{dataset}_stats.npz')
    if stats_path.exists():
        return np.load(stats_path)['pool_3'].reshape(-1, 2048)
    print(f"  Building real stats ({stats_path}) …")
    pathlib.Path('assets/stats').mkdir(parents=True, exist_ok=True)
    train_ds, _ = get_dataset(config)
    pools, total = [], 0
    for i, batch in enumerate(train_ds):
        pool = run_inception(to_uint8(np.array(batch)), inception_model)[0]
        pools.append(pool)
        total += len(pool)
        if (i + 1) % 20 == 0:
            print(f"    real stats batch {i+1}  ({total} images)")
        if total >= 50_000:
            break
    real_pool3 = np.concatenate(pools, 0)[:50_000]
    np.savez_compressed(stats_path, pool_3=real_pool3)
    print(f"  Saved {stats_path}")
    return real_pool3


# ── Qualitative plots ──────────────────────────────────────────────────────────

def plot_corrector_sweep(sde, score_fn, pred_type, corr_type, snr,
                         config, inverse_scaler, eps, out_dir, rng):
    H, C = config.data.image_size, config.data.num_channels
    N_SWEEP = 8
    corr_steps_list = [1, 2, 3, 4]
    shape_s = (N_SWEEP, H, H, C)
    print(f"Generating corrector-steps sweep ({N_SWEEP} samples, steps={corr_steps_list}) …")

    rng, fixed_k = jax.random.split(rng)
    sweep_imgs = []
    for n_corr in corr_steps_list:
        sampler = pc_sampler(sde, shape_s,
                             build_predictor(pred_type, sde, score_fn),
                             build_corrector(corr_type, sde, score_fn, snr, n_steps=n_corr),
                             inverse_scaler, denoise=True, epsilon=eps)
        imgs, _ = sampler(fixed_k)
        sweep_imgs.append(np.clip(np.array(imgs), 0, 1))
        print(f"  n_corr={n_corr} done")

    _, axes = plt.subplots(len(corr_steps_list), N_SWEEP,
                           figsize=(N_SWEEP, len(corr_steps_list)))
    for r, (n_corr, row_imgs) in enumerate(zip(corr_steps_list, sweep_imgs)):
        for c in range(N_SWEEP):
            ax = axes[r, c]
            img = row_imgs[c]
            ax.imshow(img if C == 3 else img[..., 0], cmap=None if C == 3 else 'gray')
            ax.axis('off')
        axes[r, 0].set_ylabel(f'{n_corr} step{"s" if n_corr > 1 else ""}',
                              fontsize=8, rotation=90, va='center')
    plt.subplots_adjust(wspace=0.02, hspace=0.02)
    path = out_dir / 'corrector_sweep.png'
    plt.savefig(str(path), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")
    return rng


def plot_corrector_trace(sde, score_fn, pred_type, corr_type, snr,
                         config, inverse_scaler, eps, out_dir, rng,
                         snap_steps, n_intervals):
    H, C = config.data.image_size, config.data.num_channels
    cap_anchors = sorted(set(snap_steps))
    if n_intervals is not None:
        cap_anchors = cap_anchors[-n_intervals:]
    n_snaps = len(cap_anchors)
    n_traj  = 2
    print(f"Generating corrector trace ({n_traj} samples, {n_snaps} anchors @ {cap_anchors}) …")
    shape_t = (n_traj, H, H, C)

    _corr1 = jax.jit(build_corrector(corr_type, sde, score_fn, snr, n_steps=1).update_fn)
    _pred  = jax.jit(build_predictor(pred_type, sde, score_fn).update_fn)

    rng, k = jax.random.split(rng)
    x  = sde.prior_sampling(k, shape_t)
    ts = np.linspace(sde.T, eps, sde.N)

    cap_set = set()
    for s in cap_anchors:
        cap_set.add(s)
        if s + 1 < sde.N:
            cap_set.add(s + 1)

    snaps = {}
    for i, t_val in enumerate(ts):
        vec_t = jnp.full((n_traj,), t_val)
        rng, k = jax.random.split(rng)
        x_c1, _ = _corr1(k, x, vec_t)
        rng, k = jax.random.split(rng)
        x_c2, _ = _corr1(k, x_c1, vec_t)
        rng, k = jax.random.split(rng)
        x_p,  _ = _pred(k, x_c2, vec_t)
        if i in cap_set:
            snaps[i] = (
                np.clip(np.array(inverse_scaler(x_c1)), 0, 1),
                np.clip(np.array(inverse_scaler(x_c2)), 0, 1),
                np.clip(np.array(inverse_scaler(x_p)),  0, 1),
            )
        x = x_p

    ncols  = n_snaps * 6
    canvas = np.ones((n_traj * H, ncols * H, C), dtype=np.float32)
    for j, s in enumerate(cap_anchors):
        for turn, step in enumerate([s, s + 1]):
            if step not in snaps:
                continue
            x_c1, x_c2, x_p = snaps[step]
            for row in range(n_traj):
                r0, r1 = row * H, (row + 1) * H
                c0 = (j * 6 + turn * 3) * H
                canvas[r0:r1, c0       : c0 +   H] = x_c1[row]
                canvas[r0:r1, c0 +   H : c0 + 2*H] = x_c2[row]
                canvas[r0:r1, c0 + 2*H : c0 + 3*H] = x_p[row]

    path = out_dir / 'corrector_trace.png'
    plt.imsave(str(path), canvas)
    print(f"  Saved {path}")
    print(f"  Layout: {n_traj} rows × {ncols} cols  (6 per anchor: C₁|C₂|P|C₁|C₂|P)")
    print(f"  Anchors: {cap_anchors}")
    return rng


def plot_trajectory(sde, score_fn, pred_type, corr_type, snr,
                    config, inverse_scaler, eps, out_dir, rng,
                    snap_steps, n_intervals, corrector_steps):
    H, C = config.data.image_size, config.data.num_channels
    cap_anchors = sorted(set(snap_steps))
    if n_intervals is not None:
        cap_anchors = cap_anchors[-n_intervals:]
    n_snaps = len(cap_anchors)
    n_traj  = 2
    print(f"Generating trajectory ({n_traj} samples, {n_snaps} snapshot groups @ {cap_anchors}) …")
    shape_t = (n_traj, H, H, C)

    _pred = jax.jit(build_predictor(pred_type, sde, score_fn).update_fn)
    _corr = jax.jit(build_corrector(corr_type, sde, score_fn, snr, n_steps=corrector_steps).update_fn)

    cap_set = set()
    for s in cap_anchors:
        cap_set.add(s)
        if s + 1 < sde.N:
            cap_set.add(s + 1)

    rng, k = jax.random.split(rng)
    x  = sde.prior_sampling(k, shape_t)
    ts = np.linspace(sde.T, eps, sde.N)

    snaps_pred, snaps_corr = {}, {}
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

    ncols  = n_snaps * 4
    canvas = np.ones((n_traj * H, ncols * H, C), dtype=np.float32)
    for j, s in enumerate(cap_anchors):
        for row in range(n_traj):
            r0, r1 = row * H, (row + 1) * H
            c = j * 4 * H
            canvas[r0:r1, c       : c +   H] = snaps_corr[s][row]
            canvas[r0:r1, c +   H : c + 2*H] = snaps_pred[s][row]
            if s + 1 in snaps_corr:
                canvas[r0:r1, c + 2*H : c + 3*H] = snaps_corr[s + 1][row]
                canvas[r0:r1, c + 3*H : c + 4*H] = snaps_pred[s + 1][row]

    path = out_dir / 'trajectory.png'
    plt.imsave(str(path), canvas)
    print(f"  Saved {path}")
    print(f"  Layout: {n_traj} rows × {ncols} cols")
    print(f"  Groups (step pairs): {[(s, s+1) for s in cap_anchors]}")
    return rng


def plot_perturb_recover(sde, score_fn, config, inverse_scaler, eps, out_dir, rng):
    H, C = config.data.image_size, config.data.num_channels
    print("Perturbing and recovering 20 test images via probability flow ODE …")
    N_REAL = 20

    _, eval_ds = get_dataset(config)
    scaler = get_data_scaler(config.data.centered)
    real_imgs = []
    for batch in eval_ds:
        real_imgs.append(scaler(batch).numpy())
        if sum(len(b) for b in real_imgs) >= N_REAL:
            break
    real_imgs = np.concatenate(real_imgs, axis=0)[:N_REAL]

    rng, k = jax.random.split(rng)
    x0    = jnp.array(real_imgs)
    vec_T = jnp.full((N_REAL,), sde.T)
    a_T, sig_T = sde.marginal_prob(x0, vec_T)
    x_T   = a_T + batch_mul(sig_T, jax.random.normal(k, x0.shape))

    ode_fn = ode_sampler(sde, score_fn, (N_REAL, H, H, C), inverse_scaler, eps=eps)
    rng, k = jax.random.split(rng)
    recovered, nfe = ode_fn(k, z=x_T)
    recovered = np.array(recovered).reshape(N_REAL, H, H, C)
    print(f"  ODE recovery NFE: {nfe}")

    orig_disp   = np.clip(np.array(inverse_scaler(x0)),  0, 1)
    latent_disp = np.clip(np.array(inverse_scaler(x_T)), 0, 1)

    def _row(imgs_arr):
        row = np.ones((H, N_REAL * H, C), dtype=np.float32)
        for i, img in enumerate(imgs_arr[:N_REAL]):
            row[:, i*H:(i+1)*H] = np.clip(img, 0, 1)
        return row

    canvas = np.concatenate([_row(orig_disp), _row(latent_disp), _row(recovered)], axis=0)
    path = out_dir / 'perturb_recover.png'
    plt.imsave(str(path), canvas)
    print(f"  Saved {path}  (rows: original | latent t=T | ODE recovered)")
    np.save(str(out_dir / 'latents.npy'), np.array(x_T))
    print(f"  Saved {out_dir}/latents.npy  shape={x_T.shape}")
    return rng


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    src = parser.add_mutually_exclusive_group()
    src.add_argument('--model', choices=list(MODELS_BY_NAME.keys()),
                     help='Evaluate a single named model (default: all models)')
    src.add_argument('--ckpt',
                     help='Manual checkpoint path (also requires --config)')

    parser.add_argument('--config',
                        help='Config name from config.py (required with --ckpt)')
    parser.add_argument('--sample-only', action='store_true',
                        help='Save 8×8 grid only, skip FID/IS evaluation')
    parser.add_argument('--plot',
                        choices=['corrector_sweep', 'corrector_trace', 'trajectory', 'perturb_recover'],
                        default=None,
                        help='Generate a qualitative plot (requires --model or --ckpt)')
    parser.add_argument('--snap_steps', type=int, nargs='+',
                        default=list(range(0, 1000, 100)),
                        help='PC step indices to snapshot for trajectory/corrector_trace')
    parser.add_argument('--n_intervals', type=int, default=None,
                        help='Keep only the last N snap_steps intervals')
    parser.add_argument('--corrector_steps', type=int, default=1,
                        help='Langevin corrector steps for trajectory (default: 1)')
    parser.add_argument('--seed',    type=int, default=42)
    parser.add_argument('--out_dir', default=None,
                        help='Output directory (default: assets/eval/<model>)')
    args = parser.parse_args()

    if args.ckpt and not args.config:
        parser.error('--config is required when using --ckpt')
    if args.plot and not (args.model or args.ckpt):
        parser.error('--plot requires --model or --ckpt')

    # ── Build model list ──────────────────────────────────────────────────────
    if args.ckpt:
        model_list = [(args.config, args.config, args.ckpt, False)]
    elif args.model:
        model_list = [MODELS_BY_NAME[args.model]]
    else:
        model_list = MODELS

    rng = jax.random.PRNGKey(args.seed)
    inception_model = None
    real_pool3      = None
    results         = []

    for name, cfg_name, ckpt_path, celeba_only in model_list:
        print(f"\n{'='*60}")
        print(f"Model: {name}  ({ckpt_path})")

        out_dir = (pathlib.Path(args.out_dir) if args.out_dir
                   else OUT_ROOT / name)
        out_dir.mkdir(parents=True, exist_ok=True)

        config = get_config(cfg_name)
        H, C   = config.data.image_size, config.data.num_channels
        sde, eps = get_sde(config)
        inverse_scaler = get_data_inverse_scaler(config.data.centered)

        sde_type = config.training.sde
        cont     = config.training.continuous
        pred_type, corr_type, snr = BEST_PC.get(
            (sde_type, cont), ('reverse_diffusion', 'none', 0.16)
        )
        if celeba_only:
            snr = 0.17
        print(f"  PC: predictor={pred_type}  corrector={corr_type}  snr={snr}")

        print(f"  Loading {ckpt_path} …")
        model, ema_params, _ = load_ckpt(ckpt_path, config)
        score_fn = get_score_fn(sde, model, ema_params,
                                train=False, continuous=cont)

        # ── Qualitative plot ──────────────────────────────────────────────────
        if args.plot:
            plot_kwargs = dict(
                sde=sde, score_fn=score_fn, pred_type=pred_type,
                corr_type=corr_type, snr=snr, config=config,
                inverse_scaler=inverse_scaler, eps=eps,
                out_dir=out_dir, rng=rng,
            )
            if args.plot == 'corrector_sweep':
                rng = plot_corrector_sweep(**plot_kwargs)
            elif args.plot == 'corrector_trace':
                rng = plot_corrector_trace(**plot_kwargs,
                                           snap_steps=args.snap_steps,
                                           n_intervals=args.n_intervals)
            elif args.plot == 'trajectory':
                rng = plot_trajectory(**plot_kwargs,
                                      snap_steps=args.snap_steps,
                                      n_intervals=args.n_intervals,
                                      corrector_steps=args.corrector_steps)
            elif args.plot == 'perturb_recover':
                rng = plot_perturb_recover(sde, score_fn, config,
                                           inverse_scaler, eps, out_dir, rng)
            continue

        # ── Sample-only: 8×8 grid, no FID/IS ─────────────────────────────────
        if args.sample_only or celeba_only:
            reason = "CelebA" if celeba_only else "--sample-only"
            print(f"  Generating 8×8 grid ({reason}) …")
            pred = build_predictor(pred_type, sde, score_fn)
            corr = build_corrector(corr_type, sde, score_fn, snr)
            grid_sampler = make_sampler(sde, (GRID_N, H, H, C), pred, corr, inverse_scaler, eps)
            rng, k = jax.random.split(rng)
            grid_imgs, _ = grid_sampler(k)
            save_grid(np.clip(np.array(grid_imgs), 0.0, 1.0), out_dir / 'grid.png', nrow=8)
            continue

        # ── Full eval: check cache ────────────────────────────────────────────
        result_file = out_dir / 'result.txt'
        if result_file.exists():
            parts = result_file.read_text().split()
            IS, FID, elapsed = float(parts[1]), float(parts[2]), float(parts[3])
            print(f"  Already done — IS={IS:.4f}  FID={FID:.4f}  (skipping)")
            results.append((name, IS, FID, elapsed))
            continue

        # ── Lazy-load Inception + real stats on first CIFAR model ─────────────
        if inception_model is None:
            print("Loading InceptionV1 …")
            inception_model = tfhub.load(INCEPTION_TFHUB)
            cifar_config = get_config('vpsde_ddpm_disc')
            real_pool3   = load_real_stats(cifar_config, inception_model)
            print(f"  Real pool_3: {real_pool3.shape}")

        # ── 10k samples → FID + IS + grid ────────────────────────────────────
        pred = build_predictor(pred_type, sde, score_fn)
        corr = build_corrector(corr_type, sde, score_fn, snr)

        stats_file = out_dir / 'statistics.npz'
        elapsed = 0
        if stats_file.exists():
            print("  Found cached statistics, skipping sampling …")
            cached     = np.load(stats_file)
            gen_pool3  = cached['pool_3'].reshape(-1, 2048)
            gen_logits = cached['logits']
        else:
            print(f"  Generating {N_SAMPLES} samples (batch={BATCH_SIZE}) …")
            bulk_sampler = make_sampler(sde, (BATCH_SIZE, H, H, C), pred, corr, inverse_scaler, eps)
            t0        = time.time()
            all_imgs  = []
            n_batches = int(np.ceil(N_SAMPLES / BATCH_SIZE))
            for i in range(n_batches):
                rng, k = jax.random.split(rng)
                imgs, _ = bulk_sampler(k)
                all_imgs.append(np.clip(np.array(imgs), 0.0, 1.0))
                print(f"    batch {i+1}/{n_batches}  ({time.time()-t0:.0f}s)")
            samples = np.concatenate(all_imgs, axis=0)[:N_SAMPLES]
            elapsed = time.time() - t0
            print(f"  Generated in {elapsed:.0f}s")

            save_grid(samples[:GRID_N], out_dir / 'grid.png', nrow=8)

            print("  Running Inception …")
            gen_pool3, gen_logits = run_inception(to_uint8(samples), inception_model)
            np.savez_compressed(stats_file, pool_3=gen_pool3, logits=gen_logits)

        IS  = inception_score(gen_logits)
        FID = frechet_distance(real_pool3, gen_pool3)
        print(f"  IS={IS:.4f}  FID={FID:.4f}")
        results.append((name, IS, FID, elapsed))
        result_file.write_text(f"{name}  {IS:.4f}  {FID:.4f}  {elapsed:.0f}\n")

    if results:
        print(f"\n{'═'*55}")
        print(f"{'model':<20}  {'IS':>8}  {'FID':>8}  {'time(s)':>8}")
        print(f"{'─'*55}")
        for name, IS, FID, elapsed in results:
            print(f"{name:<20}  {IS:8.4f}  {FID:8.4f}  {elapsed:8.0f}")
        print(f"{'═'*55}")


if __name__ == '__main__':
    main()
