import jax.numpy as jnp
import numpy as np

precios_np = np.array([10.0, 12.5, 9.0, 11.0])
precios = jnp.array(precios_np)
nuevos = precios.at[1].set(13.0)
print("original:", precios)
print("nuevos:  ", nuevos)
print("promedio:", float(jnp.mean(precios)), float(np.mean(precios_np)))
