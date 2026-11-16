import jax
import jax.numpy as jnp


def rendimientos(precios):
    return jnp.log(precios[1:] / precios[:-1])


print(jax.make_jaxpr(rendimientos)(jnp.linspace(100.0, 110.0, 30)))
