import argparse, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import numpy as np
import jax, jax.numpy as jnp
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import orbax.checkpoint as ocp
import tensorflow as tf
import tensorflow_hub as tfhub

tf.config.set_visible_devices([], 'GPU')

from config import get_config
from sde import VESDE, VPSDE
from model import UNet
from score import get_score_fn
from datasets import get_data_inverse_scaler, get_dataset
from samplers import (
    AncestralSamplingPredictor, ReverseDiffusionPredictor, ProbabilityFlowPredictor,
    LangevinCorrector, Predictor, Corrector,
)

INCEPTION_URL = 'https://tfhub.dev/tensorflow/tfgan/eval/inception/1'

N_SAMP = 10_000
BATCH  = 256
SNR    = 0.16

STEP_CONFIGS = [
    ('P1000',  1000, 0),
    ('P2000',  2000, 0),
    ('PC1000', 1000, 1),
]

PREDICTORS = {
    'ancestral':     lambda sde, sf: AncestralSamplingPredictor(sde, sf),
    'rev_diffusion': lambda sde, sf: ReverseDiffusionPredictor(sde, sf),
    'prob_flow':     lambda sde, sf: ProbabilityFlowPredictor(sde, sf),
}

C2000 = ('C2000', 2000, 1)

MODELS = [
    {
        'name':     'vesde_cifar10',
        'ckpt':     'runs/vesde_cifar10_disc/ckpt/190000',
        'cfg_key':  'vesde_ddpm_disc',
        'sde_type': 'vesde',
        'eps':      1e-5,
        'out':      pathlib.Path('assets/samples/vesde_sweep'),
    },
    {
        'name':     'ddpm_cifar10_low',
        'ckpt':     'runs/vpsde_cifar10_disc/ckpt/190000',
        'cfg_key':  'vpsde_ddpm_disc',
        'sde_type': 'vpsde',
        'eps':      1e-3,
        'out':      pathlib.Path('assets/samples/ddpm_sweep'),
    },
]


def make_sde(sde_type, N, config):
    if sde_type == 'vesde':
        return VESDE(config.model.sigma_min, config.model.sigma_max, N)
    factor = 2.0 if N == 2000 else 1.0
    return VPSDE(config.model.beta_min / factor, config.model.beta_max / factor, N)


@tf.function
def _inception_batch(x_uint8, model):
    x = (tf.cast(x_uint8, tf.float32) - 127.5) / 127.5
    return model(x)

def run_inception(imgs_uint8, model, bs=500):
    pools = []
    for i in range(0, len(imgs_uint8), bs):
        out = _inception_batch(tf.constant(imgs_uint8[i:i+bs]), model)
        pools.append(out['pool_3'].numpy().reshape(len(out['pool_3']), -1))
    return np.concatenate(pools, 0)

def _sym_sqrtm(m, eps=1e-10):
    s, u = np.linalg.eigh(m)
    s = np.where(s < eps, 0.0, np.sqrt(np.maximum(s, 0.0)))
    return (u * s) @ u.T

def fid(real_pool, gen_pool):
    r, g = real_pool.astype(np.float64), gen_pool.astype(np.float64)
    mu_r, mu_g = r.mean(0), g.mean(0)
    sr, sg = np.cov(r, rowvar=False), np.cov(g, rowvar=False)
    sqr = _sym_sqrtm(sr)
    tr  = np.trace(_sym_sqrtm(sqr @ sg @ sqr))
    d   = mu_r - mu_g
    return float(d @ d + np.trace(sr) + np.trace(sg) - 2.0 * tr)

def to_uint8(imgs):
    return (np.clip(imgs, 0.0, 1.0) * 255).astype(np.uint8)

def get_real_pool(config, inception_model):
    path = pathlib.Path('assets/stats/cifar10_stats.npz')
    if path.exists():
        return np.load(path)['pool_3'].reshape(-1, 2048)
    print("  Building real stats from training data …")
    path.parent.mkdir(parents=True, exist_ok=True)
    train_ds, _ = get_dataset(config)
    pools, total = [], 0
    for batch in train_ds:
        pools.append(run_inception(to_uint8(np.array(batch)), inception_model))
        total += len(pools[-1])
        if total >= 50_000:
            break
    real = np.concatenate(pools)[:50_000]
    np.savez_compressed(path, pool_3=real)
    return real


def load_model(ckpt_path, config):
    model  = UNet(config=config)
    params = ocp.PyTreeCheckpointer().restore(str(pathlib.Path(ckpt_path).resolve() / 'default'))['ema_params']
    return model, jax.device_put(params, jax.devices()[0])

def make_sampler(sde, shape, pred, corr, inv_scaler, eps):
    @jax.jit
    def _sample(rng):
        rng, k = jax.random.split(rng)
        x  = sde.prior_sampling(k, shape)
        ts = jnp.linspace(sde.T, eps, sde.N)

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

def generate_pool(sampler, rng, inception_model):
    n_batches = (N_SAMP + BATCH - 1) // BATCH
    pools = []
    t0 = time.time()
    for i in range(n_batches):
        rng, k = jax.random.split(rng)
        imgs = to_uint8(np.clip(np.array(sampler(k)), 0, 1))
        pools.append(run_inception(imgs, inception_model))
        print(f"    batch {i+1}/{n_batches}  {time.time()-t0:.0f}s", flush=True)
    return np.concatenate(pools)[:N_SAMP], rng


def run_sweep(spec, inception, real_pool, rng):
    out = spec['out']
    out.mkdir(parents=True, exist_ok=True)
    results_file = out / 'results.txt'

    config     = get_config(spec['cfg_key'])
    H, C       = config.data.image_size, config.data.num_channels
    inv_scaler = get_data_inverse_scaler(config.data.centered)
    sde_type   = spec['sde_type']
    eps        = spec['eps']

    print(f"\nLoading checkpoint: {spec['ckpt']}")
    model, params = load_model(spec['ckpt'], config)

    score_sde = make_sde(sde_type, 1000, config)
    score_fn  = get_score_fn(score_sde, model, params, train=False, continuous=False)

    results = {}
    if results_file.exists():
        for line in results_file.read_text().splitlines():
            parts = line.split()
            if len(parts) == 3:
                results[parts[0]] = (float(parts[1]), float(parts[2]))

    runs = []
    for pred_name, pred_fn in PREDICTORS.items():
        for tag, N, n_corr in STEP_CONFIGS:
            runs.append((f"{pred_name}_{tag}", N, n_corr, pred_fn))
    tag, N, n_corr = C2000
    runs.append((tag, N, n_corr, None))

    total = len(runs)
    for idx, (run_name, N, n_corr, pred_fn) in enumerate(runs, 1):
        if run_name in results:
            fid_val, elapsed = results[run_name]
            print(f"[{idx}/{total}] {run_name}  FID={fid_val:.3f}  (cached)")
            continue

        print(f"\n[{idx}/{total}] {run_name}  N={N}  n_corr={n_corr}")

        sde  = make_sde(sde_type, N, config)
        pred = pred_fn(sde, score_fn) if pred_fn is not None else Predictor()
        corr = LangevinCorrector(sde, score_fn, SNR, n_corr) if n_corr > 0 else Corrector()

        sampler = make_sampler(sde, (BATCH, H, H, C), pred, corr, inv_scaler, eps)

        rng, k = jax.random.split(rng)
        sampler(k).block_until_ready()
        print("  compiled — generating …")

        t0 = time.time()
        gen_pool, rng = generate_pool(sampler, rng, inception)
        elapsed = time.time() - t0

        fid_val = fid(real_pool, gen_pool)
        results[run_name] = (fid_val, elapsed)
        print(f"  FID={fid_val:.3f}  ({elapsed:.0f}s)")

        with open(results_file, 'a') as f:
            f.write(f"{run_name}  {fid_val:.4f}  {elapsed:.0f}\n")

    print(f"\n{'─'*45}")
    print(f"{'run':<25}  {'FID':>8}  {'time(s)':>8}")
    print(f"{'─'*45}")
    for name, (fid_val, elapsed) in sorted(results.items()):
        print(f"{name:<25}  {fid_val:8.3f}  {elapsed:8.0f}")
    print(f"{'─'*45}")
    print(f"Results saved → {results_file}")

    return rng


def draw_trajectories(spec, rng):
    SNAP_EVERY = 100

    config     = get_config(spec['cfg_key'])
    H, C       = config.data.image_size, config.data.num_channels
    inv_scaler = get_data_inverse_scaler(config.data.centered)
    sde_type   = spec['sde_type']
    eps        = spec['eps']

    print(f"Loading checkpoint: {spec['ckpt']}")
    model, params = load_model(spec['ckpt'], config)

    score_sde = make_sde(sde_type, 1000, config)
    score_fn  = get_score_fn(score_sde, model, params, train=False, continuous=False)

    sde_2000 = make_sde(sde_type, 2000, config)
    rng, k   = jax.random.split(rng)
    x0       = sde_2000.prior_sampling(k, (1, H, H, C))

    def _snap(xm):
        return np.clip(np.array(inv_scaler(xm)), 0, 1)[0]

    def run_p2000(pred_fn):
        sde  = make_sde(sde_type, 2000, config)
        _p   = jax.jit(pred_fn(sde, score_fn).update_fn)
        ts   = np.linspace(float(sde.T), eps, 2000)
        x    = x0
        snaps = []
        for i in range(2000):
            t = jnp.full((1,), ts[i])
            x, xm = _p(jax.random.fold_in(rng, i * 3), x, t)
            if (i + 1) % SNAP_EVERY == 0:
                snaps.append(_snap(xm))
        return snaps

    def run_c2000(_):
        sde  = make_sde(sde_type, 2000, config)
        _c   = jax.jit(LangevinCorrector(sde, score_fn, SNR, n_steps=1).update_fn)
        ts   = np.linspace(float(sde.T), eps, 2000)
        x    = x0
        snaps = []
        for i in range(2000):
            t = jnp.full((1,), ts[i])
            x, xm = _c(jax.random.fold_in(rng, i * 3 + 1), x, t)
            if (i + 1) % SNAP_EVERY == 0:
                snaps.append(_snap(xm))
        return snaps

    def run_pc1000(pred_fn):
        sde  = make_sde(sde_type, 1000, config)
        _c   = jax.jit(LangevinCorrector(sde, score_fn, SNR, n_steps=1).update_fn)
        _p   = jax.jit(pred_fn(sde, score_fn).update_fn)
        ts   = np.linspace(float(sde.T), eps, 1000)
        x    = x0
        snaps = []
        for i in range(1000):
            t = jnp.full((1,), ts[i])
            x, xm_c = _c(jax.random.fold_in(rng, i * 3 + 2), x, t)
            x, xm_p = _p(jax.random.fold_in(rng, i * 3 + 2 + 10000), x, t)
            if (i + 1) % SNAP_EVERY == 0:
                snaps.append(_snap(xm_c))
                snaps.append(_snap(xm_p))
        return snaps

    pred_names  = list(PREDICTORS.keys())
    sub_runners = [run_p2000, run_c2000, run_pc1000]
    sub_labels  = ['P2000', 'C2000', 'PC1000']
    N_SNAPS     = 20

    all_snaps = {}
    for pred_name, pred_fn in PREDICTORS.items():
        for label, runner in zip(sub_labels, sub_runners):
            print(f"  {pred_name} / {label} …", flush=True)
            all_snaps[(pred_name, label)] = runner(pred_fn)

    n_rows = len(pred_names) * len(sub_labels)
    fig, axes = plt.subplots(n_rows, N_SNAPS, figsize=(N_SNAPS * 1.0, n_rows * 1.1))
    for pi, pred_name in enumerate(pred_names):
        for si, label in enumerate(sub_labels):
            row   = pi * len(sub_labels) + si
            snaps = all_snaps[(pred_name, label)]
            for col, img in enumerate(snaps):
                ax = axes[row, col]
                ax.imshow(img if C == 3 else img[..., 0], cmap=None if C == 3 else 'gray', vmin=0, vmax=1)
                ax.axis('off')
            ylabel = f'{pi + 1}. {pred_name}\n{label}' if si == 0 else label
            axes[row, 0].set_ylabel(ylabel, fontsize=6, rotation=0, ha='right', va='center', labelpad=45)

    for col in range(N_SNAPS):
        axes[0, col].set_title(str((col + 1) * SNAP_EVERY), fontsize=6)

    for pi in range(len(pred_names)):
        pc_row = pi * len(sub_labels) + 2
        for col in range(N_SNAPS):
            interval = (col // 2 + 1) * SNAP_EVERY
            axes[pc_row, col].set_title(f"{'C' if col % 2 == 0 else 'P'}{interval}", fontsize=5, pad=1)

    plt.subplots_adjust(wspace=0.02, hspace=0.05)
    out_path = spec['out'].parent / f"{spec['name']}_traj.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=['vesde', 'ddpm'])
    parser.add_argument('--draw', action='store_true')
    args = parser.parse_args()

    spec = next(m for m in MODELS if m['name'].startswith(args.model))
    rng  = jax.random.PRNGKey(42)

    if args.draw:
        draw_trajectories(spec, rng)
        return

    print("Loading Inception …")
    inception = tfhub.load(INCEPTION_URL)
    real_pool = get_real_pool(get_config('vesde_ddpm_disc'), inception)
    print(f"  real pool_3: {real_pool.shape}")

    print(f"\n{'═'*50}")
    print(f"  Model: {spec['name']}")
    print(f"{'═'*50}")
    run_sweep(spec, inception, real_pool, rng)


if __name__ == '__main__':
    main()
