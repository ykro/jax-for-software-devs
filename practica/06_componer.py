import jax
import jax.numpy as jnp

lecturas = jnp.array([0.10, 0.25, 0.40, 0.55, 0.70])


def error(params, reales):
    a, b = params
    return jnp.mean((a * lecturas + b - reales) ** 2)


ruido = jax.random.normal(jax.random.key(1), (1000, 5)) * 0.05
sensores = 2.0 * lecturas + 0.1 + ruido
params = jnp.array([1.0, 0.0])

gradientes = jax.jit(jax.vmap(jax.grad(error), in_axes=(None, 0)))
g = gradientes(params, sensores)
print("un gradiente por sensor:", g.shape)
print("primeros dos:", g[:2])
