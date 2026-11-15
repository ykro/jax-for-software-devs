import jax


def puntaje(transaccion, pesos):
    return jax.nn.sigmoid(transaccion @ pesos)


k_tx, k_w = jax.random.split(jax.random.key(0))
transacciones = jax.random.normal(k_tx, (10_000, 16))
pesos = jax.random.normal(k_w, (16,))

print("una:", puntaje(transacciones[0], pesos).shape)
todas = jax.jit(jax.vmap(puntaje, in_axes=(0, None)))(transacciones, pesos)
print("todas:", todas.shape)
print("sospechosas:", int((todas > 0.99).sum()))
