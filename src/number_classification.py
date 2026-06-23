import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class NumberClassificationModel(nnx.Module):
    # 入出力層・隠れ層を定義
    def __init__(self, in_features: int, hidden_features: int, out_features: int, rngs: nnx.Rngs):
        self.layer1 = nnx.Linear(in_features, hidden_features, rngs=rngs)
        self.layer2 = nnx.Linear(hidden_features, hidden_features, rngs=rngs)
        self.layer3 = nnx.Linear(hidden_features, hidden_features, rngs=rngs)
        self.out_layer = nnx.Linear(hidden_features, out_features, rngs=rngs)

    # 順伝播を定義
    def __call__(self, x: jax.Array) -> jax.Array:
        x = jax.nn.relu(self.layer1(x))
        x = jax.nn.relu(self.layer2(x))
        x = jax.nn.relu(self.layer3(x))
        return self.out_layer(x)

def main() -> None:
    pass
