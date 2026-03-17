import jax
import jax.numpy as jnp
import tensorflow as tf
import flax

import library.models.utils as mutils
import library.sde_lib as sde_lib

from config import get_config_ddpm_cifar10, get_config_ncsnpp_cifar10, get_config_ncsnpp_bedroom, get_config_ncsnpp_celeba, get_config_ncsnpp_church, get_config_ncsnpp_ffhq_1024
from sde import get_sde, get_old_sde
import datasets
from utils import convert_params

def load_training_state(filepath, state):
  with tf.io.gfile.GFile(filepath, "rb") as f:
    state = flax.serialization.from_bytes(state, f.read())
  return state

def select_config(model, dataset, sde_type='vpsde'):
    if dataset == 'cifar10':
        if model == 'ddpm':
            return get_config_ddpm_cifar10(False, False, False, sde_type)
        elif model == 'ddpm-cont':
            return get_config_ddpm_cifar10(True, False, False, sde_type)
        elif model == 'ddpmpp':
            return get_config_ddpm_cifar10(False, True, False, sde_type)
        elif model == 'ddpmpp-cont':
            return get_config_ddpm_cifar10(True, True, False, sde_type)
        elif model == 'ddpmpp-cont-deep':
            return get_config_ddpm_cifar10(True, True, True, sde_type)
        elif model == 'ncsnpp':
            return get_config_ncsnpp_cifar10(False, False)
        elif model == 'ncsnpp-cont':
            return get_config_ncsnpp_cifar10(True, False)
        elif model == 'ncsnpp-cont-deep':
            return get_config_ncsnpp_cifar10(True, True)
    elif dataset == 'bedroom':
        return get_config_ncsnpp_bedroom()
    elif dataset == 'celeba' or dataset == 'ffhq-256':
        return get_config_ncsnpp_celeba()
    elif dataset == 'church':
        return get_config_ncsnpp_church()
    return get_config_ncsnpp_ffhq_1024()
        
    
def load_model_from_checkpoint(
        checkpoint: str,
        model: str = 'ddpm',
        dataset: str = 'cifar10',
        sde_type: str = 'vpsde',
        interpolation: str = 'linear',
        cont: bool = False
    ):
    c = select_config(model, dataset, sde_type)

    sde, sampling_eps = get_sde(c)
    sde_old = get_old_sde(c)

    sampling_eps = 1e-3

    random_seed = 0
    rng = jax.random.PRNGKey(random_seed)
    rng, run_rng = jax.random.split(rng)
    score_model, init_model_state, initial_params = mutils.init_model(run_rng, c)
    state = mutils.State(
        step=0, 
        optimizer = None,
        lr = 0,
        model_state=init_model_state,
        ema_rate=c.model.ema_rate,
        params_ema=initial_params,
        rng=rng
    )

    scaler = datasets.get_data_scaler(c.data.centered)
    inverse_scaler = datasets.get_data_inverse_scaler(c.data.centered)    
    state = load_training_state(checkpoint, state)
    fixed_params = convert_params(state.params_ema)
    cont = cont if cont else c.training.continuous
    score_func = mutils.get_score_fn(
        sde_old, 
        score_model,
        fixed_params, 
        state.model_state, 
        train=False, 
        continuous=cont
    )

    def score_fn(x,t):
        if interpolation == 'rounding':
            steps = sde.N-1
            t_round = jnp.round(t*steps)/steps
            return score_func(x, t_round, rng)
        else:
            return score_func(x,t,rng)
    
    return sde, score_fn, scaler, inverse_scaler, c, sampling_eps
