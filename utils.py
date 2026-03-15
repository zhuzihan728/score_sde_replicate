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
        for k, v in d.items():
            if isinstance(v, dict):
                convert(v)
            elif k in ['scale', 'bias'] and hasattr(v, 'ndim'):
                if v.ndim == 4 and v.shape[:3] == (1,1,1):
                    d[k] = jnp.squeeze(v)

    convert(params)
    return flax.core.freeze(params)