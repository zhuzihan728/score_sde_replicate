"""
  python exp\eval_score_sde.py --model ncsnpp
  python exp\eval_score_sde.py --model ncsnpp_deep --n_samples 1000 --batch_size 64
"""

import argparse, importlib.util, pathlib, sys, time
import numpy as np
import jax, jax.numpy as jnp
import tensorflow_hub as tfhub

_ROOT      = pathlib.Path(__file__).resolve().parent.parent
_SCORE_SDE = pathlib.Path(__file__).parent / 'score_sde'
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_SCORE_SDE))

from flax.training import checkpoints
from models import ncsnpp           # noqa: F401
from models import utils as mutils

from sde import VESDE
from samplers import ReverseDiffusionPredictor, LangevinCorrector, pc_sampler
from config import get_config
from eval import run_inception, inception_score, frechet_distance, load_real_stats, to_uint8, save_grid

INCEPTION_TFHUB = 'https://tfhub.dev/tensorflow/tfgan/eval/inception/1'

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

EPS = 1e-5
SNR = 0.16


def load_score_sde_config(config_path):
    spec = importlib.util.spec_from_file_location('_scoresde_cfg', str(config_path))
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_config()


def _fix_groupnorm_shapes(params):
    def _fix(x):
        if hasattr(x, 'shape') and x.ndim == 4 and x.shape[:3] == (1, 1, 1):
            return x.reshape((x.shape[3],))
        return x
    return jax.tree_util.tree_map(_fix, params)


def load_model(ckpt_path, config):
    rng = jax.random.PRNGKey(0)
    model, _, _ = mutils.init_model(rng, config)
    raw         = checkpoints.restore_checkpoint(ckpt_path, target=None)
    params_ema  = _fix_groupnorm_shapes(jax.device_put(raw['params_ema'],  jax.devices()[0]))
    model_state = jax.device_put(raw['model_state'], jax.devices()[0])
    print(f"  Restored checkpoint at step {int(raw['step'])}")
    return model, params_ema, model_state


def make_score_fn(sde, model, params, model_state):
    variables = {'params': params, **model_state}

    @jax.jit
    def score_fn(x, t):
        sigma = sde.marginal_prob(jnp.zeros_like(x), t)[1]
        return model.apply(variables, x, sigma, train=False, mutable=False)

    return score_fn


def run_sampling(sampler_fn, n_samples, batch_size, rng):
    imgs_list = []
    n_batches = int(np.ceil(n_samples / batch_size))
    t0 = time.time()
    for i in range(n_batches):
        rng, k = jax.random.split(rng)
        batch, _ = sampler_fn(k)
        imgs_list.append(np.clip(np.array(batch), 0.0, 1.0))
        print(f"  batch {i + 1}/{n_batches}  ({time.time() - t0:.1f}s)")
    return np.concatenate(imgs_list, 0)[:n_samples]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',      choices=list(MODELS), default='ncsnpp')
    parser.add_argument('--n_samples',  type=int, default=10_000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed',       type=int, default=0)
    parser.add_argument('--out_dir',    default=None)
    args = parser.parse_args()

    ckpt_path, cfg_path = MODELS[args.model]
    out_dir = pathlib.Path(args.out_dir or f'assets/eval_ncsnpp/{args.model}')
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Evaluating: {args.model} ===")
    print(f"  ckpt    : {ckpt_path}")
    print(f"  config  : {cfg_path}")
    print(f"  samples : {args.n_samples}  batch_size: {args.batch_size}")

    cfg = load_score_sde_config(cfg_path)
    sde = VESDE(sigma_min=cfg.model.sigma_min, sigma_max=cfg.model.sigma_max, N=cfg.model.num_scales)
    inverse_scaler = (lambda x: (x + 1.) / 2.) if cfg.data.centered else (lambda x: x)

    print("Loading checkpoint …")
    model, params_ema, model_state = load_model(ckpt_path, cfg)
    score_fn = make_score_fn(sde, model, params_ema, model_state)

    H, C  = cfg.data.image_size, cfg.data.num_channels
    shape = (args.batch_size, H, H, C)
    pred    = ReverseDiffusionPredictor(sde, score_fn, probability_flow=False)
    corr    = LangevinCorrector(sde, score_fn, snr=SNR, n_steps=1)
    sampler = pc_sampler(sde, shape, pred, corr, inverse_scaler, n_steps=1, denoise=True, epsilon=EPS)

    rng = jax.random.PRNGKey(args.seed)
    print(f"\nGenerating {args.n_samples} samples …")
    t_gen = time.time()
    samples = run_sampling(sampler, args.n_samples, args.batch_size, rng)
    t_gen = time.time() - t_gen
    print(f"  Generated in {t_gen:.1f}s  range=[{samples.min():.3f}, {samples.max():.3f}]")

    save_grid(samples[:64], out_dir / 'grid.png', nrow=8)
    np.savez_compressed(out_dir / 'samples.npz', samples=samples)

    print("Loading InceptionV1 …")
    inception_model = tfhub.load(INCEPTION_TFHUB)

    print("Running Inception on generated samples …")
    gen_pool3, gen_logits = run_inception(to_uint8(samples), inception_model, batch_size=256)
    np.savez_compressed(out_dir / 'inception_activations.npz', pool3=gen_pool3, logits=gen_logits)

    print("Loading real CIFAR-10 stats …")
    real_pool3 = load_real_stats(get_config('vpsde_ddpm_disc'), inception_model)

    IS  = inception_score(gen_logits)
    FID = frechet_distance(real_pool3, gen_pool3)

    print(f"\n{'=' * 50}")
    print(f"  Model    : {args.model}")
    print(f"  Samples  : {args.n_samples}")
    print(f"  IS       : {IS:.4f}")
    print(f"  FID      : {FID:.4f}")
    print(f"  Gen time : {t_gen:.1f}s")
    print(f"{'=' * 50}")

    result = f"model={args.model}  n={args.n_samples}  IS={IS:.4f}  FID={FID:.4f}  t={t_gen:.0f}s\n"
    (out_dir / 'result.txt').write_text(result)
    print(f"Saved → {out_dir}")


if __name__ == '__main__':
    main()
