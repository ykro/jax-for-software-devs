import jax
import jax.numpy as jnp


@jax.jit
def alerta(rendimientos):
    if rendimientos.mean() < 0:
        return jnp.zeros_like(rendimientos)
    return rendimientos


print(alerta(jnp.array([0.01, -0.03, 0.04])))
