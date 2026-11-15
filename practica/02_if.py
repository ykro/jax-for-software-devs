import jax
import jax.numpy as jnp


@jax.jit
def alerta(rendimientos):
    return jnp.where(rendimientos.mean() < 0, jnp.zeros_like(rendimientos), rendimientos)


print(alerta(jnp.array([0.01, -0.03, 0.04])))
print(alerta(jnp.array([0.01, -0.03, -0.04])))
