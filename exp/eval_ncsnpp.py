"""
  python eval_ncsnpp.py --model ncsnpp
  python eval_ncsnpp.py --model ncsnpp_deep --n_samples 1000 --batch_size 64
"""

import argparse, importlib.util, pathlib, sys, time
import numpy as np
import jax, jax.numpy as jnp
import tensorflow_hub as tfhub

# ── Ensure MLMI4/ root and score_sde/ are both on the path ────────────────────
_ROOT      = pathlib.Path(__file__).resolve().parent.parent
_SCORE_SDE = pathlib.Path(__file__).parent / 'score_sde'
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_SCORE_SDE))

from flax.training import checkpoints
from models import ncsnpp           # noqa: F401  registers 'ncsnpp' in mutils._MODELS
from models import utils as mutils

# ── Our reimplementations ──────────────────────────────────────────────────────
from sde import VESDE
from samplers import ReverseDiffusionPredictor, LangevinCorrector, pc_sampler
from config import get_config

# ── FID / IS helpers (verified correct in eval.py) ────────────────────────────
from eval import (
    run_inception,
    inception_score, frechet_distance,
    load_real_stats, to_uint8, save_grid,
)

INCEPTION_TFHUB = 'https://tfhub.dev/tensorflow/tfgan/eval/inception/1'

def get_inception_model():
    return tfhub.load(INCEPTION_TFHUB)

# ══════════════════════════════════════════════════════════════════════════════
MODELS = {
    'ncsnpp': (
        'score_sde/ckpts/ve_cifar10_ncsnpp_continuous',
        _SCORE_SDE / 'configs/ve/cifar10_ncsnpp_continuous.py',
    ),
    'ncsnpp_deep': (
        'score_sde/ckpts/ve_cifar10_ncsnpp_deep_continuous',
        _SCORE_SDE / 'configs/ve/cifar10_ncsnpp_deep_continuous.py',
    ),
}

EPS = 1e-5   # VESDE reverse-time lower bound (t_min)
SNR = 0.16   # Langevin corrector signal-to-noise ratio (paper Table 1)
# ══════════════════════════════════════════════════════════════════════════════


def load_score_sde_config(config_path):
    """Import a score_sde config .py and return its ml_collections.ConfigDict."""
    spec = importlib.util.spec_from_file_location('_scoresde_cfg', str(config_path))
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_config()


def _fix_groupnorm_shapes(params):
    """Reshape GroupNorm scale/bias from old Flax (1,1,1,C) to new Flax (C,).

    Old Flax stored GroupNorm parameters as (1,1,1,C); modern Flax expects (C,).
    This only touches tensors with exactly that shape, leaving all other
    parameters (conv kernels, dense weights, etc.) untouched.
    """
    def _fix(x):
        if hasattr(x, 'shape') and x.ndim == 4 and x.shape[:3] == (1, 1, 1):
            return x.reshape((x.shape[3],))
        return x
    return jax.tree_util.tree_map(_fix, params)


def load_model(ckpt_path, config):
    """
    Initialise NCSNPP from config and restore EMA weights.

    score_sde checkpoints are flax msgpack files (not orbax).  We use
    flax.training.checkpoints.restore_checkpoint with target=None to get a
    raw dict, then pull out params_ema and model_state.
    """
    rng = jax.random.PRNGKey(0)
    model, _, _ = mutils.init_model(rng, config)          # architecture only
    raw   = checkpoints.restore_checkpoint(ckpt_path, target=None)
    params_ema  = jax.device_put(raw['params_ema'],  jax.devices()[0])
    model_state = jax.device_put(raw['model_state'], jax.devices()[0])
    params_ema  = _fix_groupnorm_shapes(params_ema)
    step        = int(raw['step'])
    print(f"  Restored checkpoint at step {step}")
    return model, params_ema, model_state


def make_score_fn(sde, model, params, model_state):
    """
    Build score(x, t) → ∇_x log p_t(x) for a continuous VESDE NCSNPP.

    Why this wrapper (not our score.py):
      - NCSNPP with embedding_type='fourier' expects time_cond = σ(t)  (raw sigma).
        It takes log(sigma) internally for Gaussian Fourier embedding.
      - With scale_by_sigma=True the model already outputs h/σ, which IS the
        score — no further 1/σ scaling is needed.
      Our score.py passes log(σ) and also divides by σ externally → double error.

    Reference: score_sde/models/utils.py  get_score_fn  VESDE continuous branch.
    """
    variables = {'params': params, **model_state}

    @jax.jit
    def score_fn(x, t):
        # sigma(t): shape (B,) — marginal_prob ignores x for VESDE
        sigma = sde.marginal_prob(jnp.zeros_like(x), t)[1]
        # model.apply returns the score directly (scale_by_sigma done inside)
        return model.apply(variables, x, sigma, train=False, mutable=False)

    return score_fn


def run_sampling(sampler_fn, n_samples, batch_size, rng):
    """Collect n_samples images by repeated calls to sampler_fn."""
    imgs_list = []
    n_batches = int(np.ceil(n_samples / batch_size))
    t0 = time.time()
    for i in range(n_batches):
        rng, k = jax.random.split(rng)
        batch, _ = sampler_fn(k)
        imgs_list.append(np.clip(np.array(batch), 0.0, 1.0))
        print(f"  batch {i + 1}/{n_batches}  ({time.time() - t0:.1f}s)")
    return np.concatenate(imgs_list, 0)[:n_samples]


# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Eval score_sde NCSNPP: IS + FID')
    parser.add_argument('--model',      choices=list(MODELS), default='ncsnpp',
                        help='ncsnpp or ncsnpp_deep')
    parser.add_argument('--n_samples',  type=int, default=10_000,
                        help='Number of generated samples for evaluation')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size per JIT call (single GPU)')
    parser.add_argument('--seed',       type=int, default=0)
    parser.add_argument('--out_dir',    default=None,
                        help='Output directory (default: assets/eval_ncsnpp/<model>)')
    args = parser.parse_args()

    ckpt_path, cfg_path = MODELS[args.model]
    out_dir = pathlib.Path(args.out_dir or f'assets/eval_ncsnpp/{args.model}')
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Evaluating: {args.model} ===")
    print(f"  ckpt    : {ckpt_path}")
    print(f"  config  : {cfg_path}")
    print(f"  samples : {args.n_samples}  batch_size: {args.batch_size}")

    # ── 1. Config ─────────────────────────────────────────────────────────────
    cfg = load_score_sde_config(cfg_path)

    # ── 2. Build our VESDE ────────────────────────────────────────────────────
    # sigma_min=0.01, sigma_max=50, N=1000 for both NCSNPP configs
    sde = VESDE(
        sigma_min=cfg.model.sigma_min,
        sigma_max=cfg.model.sigma_max,
        N=cfg.model.num_scales,
    )
    # centered=False → identity; centered=True → (x+1)/2
    inverse_scaler = (lambda x: (x + 1.) / 2.) if cfg.data.centered else (lambda x: x)

    # ── 3. Load model and build score function ────────────────────────────────
    print("Loading checkpoint …")
    model, params_ema, model_state = load_model(ckpt_path, cfg)
    score_fn = make_score_fn(sde, model, params_ema, model_state)
    print("  Score function ready.")

    # ── 4. PC sampler: reverse_diffusion + langevin (paper Table 1 for VESDE) ─
    H, C  = cfg.data.image_size, cfg.data.num_channels
    shape = (args.batch_size, H, H, C)

    pred    = ReverseDiffusionPredictor(sde, score_fn, probability_flow=False)
    corr    = LangevinCorrector(sde, score_fn, snr=SNR, n_steps=1)
    sampler = pc_sampler(
        sde, shape, pred, corr, inverse_scaler,
        n_steps=sde.N, denoise=True, epsilon=EPS,
    )

    # ── 5. Generate samples ───────────────────────────────────────────────────
    rng = jax.random.PRNGKey(args.seed)
    print(f"\nGenerating {args.n_samples} samples …")
    t_gen = time.time()
    samples = run_sampling(sampler, args.n_samples, args.batch_size, rng)
    t_gen = time.time() - t_gen
    print(f"  Generated in {t_gen:.1f}s  "
          f"range=[{samples.min():.3f}, {samples.max():.3f}]")

    # Visual sanity check: save a 8×8 grid of the first 64 samples
    save_grid(samples[:64], out_dir / 'grid.png', nrow=8)

    # Save raw samples so they survive if the Inception/FID step fails
    samples_path = out_dir / 'samples.npz'
    np.savez_compressed(samples_path, samples=samples)
    print(f"  Saved samples → {samples_path}")

    # ── 6. FID + IS via InceptionV1 ───────────────────────────────────────────
    print("Loading InceptionV1 …")
    inception_model = get_inception_model()

    print("Running Inception on generated samples …")
    gen_pool3, gen_logits = run_inception(
        to_uint8(samples), inception_model, batch_size=256
    )
    np.savez_compressed(out_dir / 'inception_activations.npz',
                        pool3=gen_pool3, logits=gen_logits)

    # Real CIFAR-10 pool_3 activations (50k); cached after first run
    print("Loading real CIFAR-10 stats …")
    cifar_cfg  = get_config('vpsde_ddpm_disc')   # any CIFAR-10 config works
    real_pool3 = load_real_stats(cifar_cfg, inception_model)

    IS  = inception_score(gen_logits)
    FID = frechet_distance(real_pool3, gen_pool3)

    # ── 7. Report ─────────────────────────────────────────────────────────────
    print(f"\n{'=' * 50}")
    print(f"  Model    : {args.model}")
    print(f"  Samples  : {args.n_samples}")
    print(f"  IS       : {IS:.4f}")
    print(f"  FID      : {FID:.4f}")
    print(f"  Gen time : {t_gen:.1f}s")
    print(f"{'=' * 50}")

    result = (f"model={args.model}  n={args.n_samples}  "
              f"IS={IS:.4f}  FID={FID:.4f}  t={t_gen:.0f}s\n")
    (out_dir / 'result.txt').write_text(result)
    print(f"Saved → {out_dir}")


if __name__ == '__main__':
    main()
