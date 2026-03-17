import jax
import flax
import jax.numpy as jnp
import numpy as np
from scipy import integrate

def get_likelihood_fn(
        sde, score_fn, inverse_scaler, hutchinson='Rademacher',
        rtol=1e-5, atol=1e-5, method='RK45', eps=1e-5
    ):

    num_devices = jax.local_device_count()

    @jax.pmap
    def drift_and_div(x, t, epsilon):
        def drift_fn(x):
            f, _  = sde.reverse_sde(x, t, score_fn(x,t), probability_flow=True)
            return f
        
        drift, jvp = jax.jvp(drift_fn, (x,), (epsilon,))
        axes = tuple(range(1,len(x.shape)))
        return drift, jnp.sum(jvp*epsilon, axis=axes)
    
    @jax.pmap
    def get_prior_logp(z):
        return sde.prior_logp(z)
    
    def likelihood_fn(rng, data):
        global_batch_size = data.shape[0]
        local_batch_size = global_batch_size // num_devices
        shape = data.shape[1:]
        multi_device_shape = (num_devices, local_batch_size)+shape
        data_reshaped = data.reshape(multi_device_shape)
        rng, step_rng = jax.random.split(rng)

        epsilon = jax.random.normal(step_rng, multi_device_shape) if hutchinson=='Gaussian' else jax.random.randint(
            step_rng, multi_device_shape, minval=0, maxval=2).astype(jnp.float32)*2-1
        
        def ode_fn(t, x_flat):
            x_reshaped = x_flat[:-global_batch_size].reshape(multi_device_shape)
            vec_t = jnp.full((num_devices,local_batch_size), t)
            drift, div = drift_and_div(x_reshaped, vec_t, epsilon)
            # drift,div = -drift, -div
            return np.concatenate([np.array(drift).ravel(), np.array(div).ravel()], axis=0)
        
        init = np.concatenate([np.array(data_reshaped).ravel(), np.zeros((global_batch_size,))], axis=0)
        solution = integrate.solve_ivp(ode_fn, (eps, sde.T), 
                        init, rtol=rtol, atol=atol, method=method
                    )
        nfe = solution.nfev
        final = solution.y[:, -1]
        z_flat = final[:-global_batch_size]
        delta_logp = final[-global_batch_size:]
        z = z_flat.reshape(multi_device_shape)
        prior_logp = get_prior_logp(z)
        prior_logp = np.array(prior_logp).ravel()
        log_likelihood = prior_logp+delta_logp
        n_dims = np.prod(shape)
        bpd = -(log_likelihood/np.log(2.0))/n_dims
        offset = jnp.log2(jax.grad(inverse_scaler)(0.))+8.
        bpd += offset

        # z = x(T): Uniquely Identifiable Representation (see paper, Section 4.3)
        return log_likelihood, bpd, z, nfe
    
    return likelihood_fn
