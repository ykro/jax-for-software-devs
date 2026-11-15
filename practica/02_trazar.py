import jax
import jax.numpy as jnp


@jax.jit
def riesgo(precios):
    print("trazando con", precios.shape, precios.dtype)
    rendimientos = jnp.diff(jnp.log(precios))
    return jnp.std(rendimientos)


dia = jnp.linspace(100.0, 110.0, 30)
riesgo(dia)
riesgo(dia * 1.01)
riesgo(dia * 0.99)
print("tres llamadas listas")
riesgo(jnp.linspace(100.0, 110.0, 60))
