import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax as dx
from jaxtyping import Float, Array, Int, PRNGKeyArray

class CNNEmulator(eqx.Module):
    layers: list

    def __init__(self, key: PRNGKeyArray, hidden_dim: Int = 4):
        self.layers = []
        self.layers.append(eqx.nn.Conv2d(2, hidden_dim, 3, padding=[1,1], key = key))
        self.layers.append(eqx.nn.Lambda(jnp.tanh))
        for i in range(4):
            key, subkey = jax.random.split(key)
            # Take two frames then predict the next one
            self.layers.append(eqx.nn.Conv2d(hidden_dim, hidden_dim, 3, padding=[1,1], key = subkey))

            self.layers.append(eqx.nn.Lambda(jnp.tanh))
        key, subkey = jax.random.split(key)
        self.layers.append(eqx.nn.Conv2d(hidden_dim, 1, 3, padding=[1,1], key = key))

    def __call__(self, x: Float[Array, "2 n_res n_res"]) -> Float[Array, "1 n_res n_res"]:
        for layer in self.layers:
            x = layer(x)
        return x
