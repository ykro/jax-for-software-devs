import jax

llave = jax.random.key(42)
a = jax.random.normal(llave, (3,))
b = jax.random.normal(llave, (3,))
print("misma llave:", a, b)

k1, k2, k_workers = jax.random.split(llave, 3)
print("separadas:  ", jax.random.normal(k1, (3,)), jax.random.normal(k2, (3,)))

workers = jax.random.split(k_workers, 4)
sorteos = jax.vmap(lambda k: jax.random.uniform(k))(workers)
print("4 workers:  ", sorteos)
