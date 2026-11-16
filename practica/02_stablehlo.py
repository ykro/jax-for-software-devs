import jax
import jax.numpy as jnp


def rendimientos(precios):
    return jnp.log(precios[1:] / precios[:-1])


print(jax.jit(rendimientos).lower(jnp.linspace(100.0, 110.0, 30)).as_text())
