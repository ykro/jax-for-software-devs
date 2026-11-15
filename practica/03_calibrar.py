import jax
import jax.numpy as jnp

lecturas = jnp.array([0.10, 0.25, 0.40, 0.55, 0.70])
reales = jnp.array([0.31, 0.62, 0.90, 1.22, 1.49])


def error(params):
    a, b = params
    return jnp.mean((a * lecturas + b - reales) ** 2)


grad_error = jax.grad(error)
params = jnp.array([1.0, 0.0])
print("gradiente:", grad_error(params))

paso = jax.jit(lambda p: p - 0.5 * grad_error(p))
for i in range(301):
    if i % 100 == 0:
        print(f"paso {i:3d}  error {error(params):.5f}  a={params[0]:.3f} b={params[1]:.3f}")
    params = paso(params)
