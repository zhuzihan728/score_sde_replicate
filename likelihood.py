import jax
import flax
import jax.numpy as jnp
from scipy import integrate

def get_likelihood_fn(
        sde, score_fn, inverse_scaler, hutchinson='Rademacher',
        rtol=1e-5, atol=1e-5, method='RK45', eps=1e-5
    ):
    @jax.jit
    def drift_and_div(x, t, epsilon):
        def drift_fn(x):
            f, _  = sde.reverse_sde(x, t, score_fn(x,t), probability_flow=True)
            return f
        
        drift, jvp = jax.jvp(drift_fn, (x,), (epsilon,))
        axes = tuple(range(1,len(x.shape)))
        return drift, jnp.sum(jvp*epsilon, axis=axes)
    
    def likelihood_fn(rng, data):
        rng, step_rng = jax.random.split(rng)
        shape = data.shape

        epsilon = jax.random.normal(step_rng, shape) if hutchinson=='Gaussian' else jax.random.randint(
            step_rng, shape, minval=0, maxval=2).astype(jnp.float32)*2-1
        
        def ode_fn(t, x_flat):
            x_reshaped = x_flat[:-shape[0]].reshape(shape)
            vec_t = jnp.full((shape[0],), t)
            drift, div = drift_and_div(x_reshaped, vec_t, epsilon)
            return jnp.concatenate([jnp.ravel(drift), jnp.ravel(div)], axis=0)
        
        init = jnp.concatenate([jnp.ravel(data), jnp.zeros((shape[0],))], axis=0)
        solution = integrate.solve_ivp(ode_fn, (eps, sde.T), init, rtol=rtol, atol=atol, method=method)
        nfe = solution.nfev
        final = solution.y[:,-1]
        z_flat = final[:-shape[0]]
        delta_logp = final[-shape[0]:]
        z = z_flat.reshape(shape)
        prior_logp = sde.prior_logp(z)
        log_likelihood = prior_logp+delta_logp

        n_dims = jnp.prod(jnp.array(shape[1:]))
        # Likelihood in bits/dimension
        bpd = -(log_likelihood/jnp.log(2.0))/n_dims
        # z = x(T): Uniquely Identifiable Representation (see paper, Section 4.3)
        return log_likelihood, bpd, z, nfe
    
    return likelihood_fn
