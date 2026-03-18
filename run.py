import warnings

warnings.filterwarnings('ignore')

import argparse

import os
import shutil

nvcc_path = shutil.which('nvcc')
cuda_dir = os.path.dirname(os.path.dirname(nvcc_path))
os.environ['XLA_FLAGS'] = f"--xla_gpu_cuda_data_dir={cuda_dir}"
print('Connected JAX to Cuda')
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.70'

import tensorflow as tf

import numpy as np

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print('Connected Tensorflow to Cuda')
    except RuntimeError as e:
        print('Failed to connect Tensorflow to Cuda')
else:
    print('Tensorflow found no GPUs')

import jax
import jax.numpy as jnp

# imports in order to register models
import library.models.ddpm
import library.models.ncsnpp

from load import load_model_from_checkpoint
from samplers import get_sampler
from evaluation import evaluation_metrics, load_dataset_stats, get_inception_model, run_inception
from likelihood import get_likelihood_fn
from datasets import get_dataset

parser = argparse.ArgumentParser(description='Sampling Experiments')

parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--experiment_type', type=str, default='generation')
parser.add_argument('--checkpoint', type=str, default='checkpoints/vp/cifar10_ddpm/checkpoint_26')
parser.add_argument('--model', type=str, default='ddpm')
parser.add_argument('--save_name', type=str, default='P1000')
parser.add_argument('--sde', type=str, default='vpsde')
parser.add_argument('--random_key', type=int, default=0)
# Sampling Parameters
parser.add_argument('--sampler', type=str, default='pc')
parser.add_argument('--predictor', type=str, default='Euler-Maruyama')
parser.add_argument('--corrector', type=str, default='AnnealedLangevin')
parser.add_argument('--snr', type=float, default=0.1)
parser.add_argument('--sampler_steps', type=int, default=1000)
parser.add_argument('--interpolation', type=str, default='linear')
parser.add_argument('--corrector_steps', type=int, default=1)
parser.add_argument('--num_gen_batches', type=int, default=1)
parser.add_argument('--inception_batch_size', type=int, default=64)
# Likelihood Parameters
parser.add_argument('--num_iters', type=int, default=5)
parser.add_argument('--hutchinson', type=str, default='Rademacher')
parser.add_argument('--max_images', type=int, default=1000)
args = parser.parse_args()

rng = jax.random.PRNGKey(args.random_key)

print(f'Loading Model from {args.checkpoint}')
cont = True if args.experiment_type == 'likelihood' else False
sde, score_fn, scaler, inverse_scaler, config, _ = load_model_from_checkpoint(
    args.checkpoint, args.model, args.dataset, args.sde, args.interpolation, cont
)

num_devices = jax.local_device_count()
print(f'JAX has access to {num_devices} GPU(s)')

if args.experiment_type == 'generation':
    config.sampler.type = args.sampler
    config.sampler.predictor = args.predictor
    config.sampler.corrector = args.corrector
    config.sampler.corrector_snr = args.snr
    config.sampler.sampler_steps = args.sampler_steps
    config.sampler.corrector_steps = args.corrector_steps

    print('Building Sampler')
    sampler_fn = get_sampler(sde, score_fn, config)

    gen_img = []

    print('\nGeneration')
    for i in range(args.num_gen_batches):
        print(f"Batch {i+1}/{args.num_gen_batches}")
        if args.sampler == 'pc':
            rng, *step_rngs = jax.random.split(rng, num_devices+1)
            step_rngs = jnp.stack(step_rngs)
            gen, _ = sampler_fn(step_rngs)
        else:
            rng, step_rng = jax.random.split(rng)
            gen, _ = sampler_fn(step_rng)

        gen = gen.reshape((-1,)+gen.shape[2:])
        gen = jnp.clip(gen*255, 0, 255).astype(jnp.uint8)
        gen_img.append(gen)

    gen_img = jnp.concatenate(gen_img, axis=0)
    gen_img_tf = tf.convert_to_tensor(gen_img)

    print('\nEvaluation\n')
    real_stats = load_dataset_stats(config)

    inception_model = get_inception_model()
    inception_batches = len(gen_img) // args.inception_batch_size
    gen_stats = run_inception(
        gen_img_tf, inception_model, num_batches=inception_batches
    )
    gen_stats = {
        'pool_3': gen_stats['pool_3'].numpy(),
        'logits': gen_stats['logits'].numpy()
    }

    is_mean, is_std, fid = evaluation_metrics(real_stats, gen_stats)
    print(f'Metrics:\nIS: {is_mean}+-{is_std}\nFID: {fid}')

    save_path = f'generated/{args.save_name}.npz'
    np.savez_compressed(
        save_path,
        images=gen_img,
        pool_3=gen_stats['pool_3'],
        logits=gen_stats['logits'],
    )
    print(f'\nGenerated images, FID and IS features saved to {save_path}')

else: # Likelihood Computation
    likelihood_fn = get_likelihood_fn(sde, score_fn, inverse_scaler, args.hutchinson, eps=1e-5)
    print(f'Loading {args.dataset} data')
    _, eval_ds = get_dataset(config)
    eval_iter = eval_ds.as_numpy_iterator()
    all_bpd = []
    processed = 0
    print('Likelihood Computation')
    for idx, images in enumerate(eval_iter):
        scaled = scaler(images)
        batch_bpd = []
        for i in range(args.num_iters):
            rng, step_rng = jax.random.split(rng)
            _, bpd, _, _ = likelihood_fn(step_rng, scaled)
            batch_bpd.append(np.mean(bpd))

        batch_avg = np.mean(batch_bpd)
        all_bpd.append(batch_avg)

        processed += images.shape[0]
        if processed >= args.max_images:
            break

    bpd = np.mean(all_bpd)
    print(f'\nNegative Log-Likelihood (NLL): {bpd:.4f} (Bits/Dim)')