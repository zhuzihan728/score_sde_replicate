import flax

import jax
import jax.numpy as jnp

def batch_mul(a, b):
    """Element-wise multiply a (batch,) with b (batch, ...) via vmap."""
    return jax.vmap(lambda a, b: a * b)(a, b)

def convert_params(params):
    """Convert parameter format to adjust to the authors' score function"""
    if isinstance(params, flax.core.FrozenDict):
        params = params.unfreeze()

    def convert(d):
        is_norm_layer = 'scale' in d
        for k, v in d.items():
            if isinstance(v, dict) or isinstance(v, flax.core.FrozenDict):
                convert(v)
            elif k in ['scale', 'bias'] and hasattr(v, 'ndim'):
                if v.ndim == 1 and is_norm_layer:
                    d[k] = jnp.reshape(v, (1,1,1,-1))

    convert(params)
    return flax.core.freeze(params)