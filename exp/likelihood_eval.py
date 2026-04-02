"""
python exp/likelihood_eval.py \
    --ckpt runs/vpsde_cifar10_cont/ckpt/190000 \
    --config vpsde_ddpm_cont
"""

import sys, pathlib, argparse
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import jax, jax.numpy as jnp
import orbax.checkpoint as ocp

from config import get_config
from sde import get_sde
from model import UNet
from score import get_score_fn
from datasets import get_data_scaler, get_data_inverse_scaler, get_dataset
from likelihood import get_likelihood_fn


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',       required=True)
    p.add_argument('--config',     required=True)
    p.add_argument('--max_images', type=int,   default=1000)
    p.add_argument('--num_iters',  type=int,   default=5,
                   help='Hutchinson estimates per batch to average')
    p.add_argument('--hutchinson', default='Rademacher',
                   choices=['Rademacher', 'Gaussian'])
    p.add_argument('--seed',       type=int,   default=42)
    return p.parse_args()


def main():
    args   = parse_args()
    config = get_config(args.config)
    sde, eps = get_sde(config)
    scaler         = get_data_scaler(config.data.centered)
    inverse_scaler = get_data_inverse_scaler(config.data.centered)

    H, C = config.data.image_size, config.data.num_channels
    model = UNet(config=config)
    model.init(jax.random.PRNGKey(0), jnp.ones((1, H, H, C)), jnp.ones(1))
    restored   = ocp.PyTreeCheckpointer().restore(
        str(pathlib.Path(args.ckpt).resolve() / 'default'))
    ema_params = jax.device_put(restored['ema_params'], jax.devices()[0])

    score_fn = get_score_fn(sde, model, ema_params, train=False,
                            continuous=config.training.continuous)
    likelihood_fn = get_likelihood_fn(sde, score_fn, inverse_scaler,
                                      hutchinson=args.hutchinson, eps=eps)

    _, eval_ds = get_dataset(config)

    rng     = jax.random.PRNGKey(args.seed)
    all_bpd = []
    processed = 0

    print(f'Computing BPD on {args.max_images} images ({args.num_iters} Hutchinson iters/batch)')
    for images in eval_ds.as_numpy_iterator():
        scaled = scaler(images)

        iter_bpds = []
        for _ in range(args.num_iters):
            rng, step_rng = jax.random.split(rng)
            _, bpd, _, _ = likelihood_fn(step_rng, scaled)
            iter_bpds.append(np.array(bpd))

        per_image = np.mean(iter_bpds, axis=0)
        all_bpd.extend(per_image.tolist())
        processed += images.shape[0]
        print(f'  {processed}/{args.max_images}  batch BPD={per_image.mean():.4f}')

        if processed >= args.max_images:
            break

    bpds = np.array(all_bpd[:args.max_images])
    print(f'\nNLL: {bpds.mean():.4f} ± {bpds.std():.4f} bits/dim')


if __name__ == '__main__':
    main()
