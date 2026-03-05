import jax

def batch_mul(a, b):
    """Element-wise multiply a (batch,) with b (batch, ...) via vmap."""
    return jax.vmap(lambda a, b: a * b)(a, b)