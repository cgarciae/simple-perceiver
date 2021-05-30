import os
from datetime import datetime
import math
from typing import Any, Generator, Mapping, Tuple

import dataget

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from tensorboardX.writer import SummaryWriter
import typer
import optax
import einops

import elegy

class Perceiver(elegy.Module):
    """Standar Perceiver implemented in live code."""

    def __init__(
        self,
        size: int,
        num_layers: int,
        reps_per_layer: int,
        num_heads: int,
        dropout: float,
        n_latents: int,
        output_dim: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.size = size
        self.num_layers = num_layers
        self.reps_per_layer = reps_per_layer
        self.num_heads = num_heads
        self.dropout = dropout
        self.n_latents = n_latents
        self.output_dim = output_dim

    def call(self, x: jnp.ndarray) -> jnp.ndarray:
        x = FourierFeatureEncoding(10, 6)(x[..., None])
        x = einops.rearrange(x, 'b ... d -> b (...) d')

        batch_size = x.shape[0]
        n_channels = x.shape[-1]

        # create latent queries
        latent = self.get_embeddings("latent", batch_size, (self.n_latents, n_channels))

        for _ in range(self.num_layers):
            block = PerceiverBlock(self.size, self.num_heads, self.dropout)

            for _ in range(self.reps_per_layer):
                latent = block(latent, x)

        # get predict output token
        # latent = latent[:, 0]
        latent = jnp.mean(latent, axis=1)

        # apply predict head
        logits = elegy.nn.Linear(self.output_dim)(latent)

        return logits

    def get_embeddings(
        self, name: str, batch_size: int, shape: Tuple[int]
    ) -> jnp.ndarray:

        embeddings = self.add_parameter(
            f"{name}_embeddings",
            lambda: elegy.initializers.TruncatedNormal()(shape, jnp.float32),
        )
        embeddings = einops.repeat(
            embeddings,
            "... -> batch ...",
            batch=batch_size,
        )

        return embeddings


class PerceiverBlock(elegy.Module):
    def __init__(
        self,
        size: int,
        num_heads: int,
        dropout: float,
    ):
        super().__init__()
        self.size = size
        self.num_heads = num_heads
        self.dropout = dropout

    def call(self, latent, x):

        latent += self.norm(self.cross_attn(latent, x))
        latent += self.norm(self.mlp(latent))
        latent += self.norm(self.self_attn(latent))
        latent += self.norm(self.mlp(latent))

        return latent

    def cross_attn(self, query: jnp.ndarray, key: jnp.ndarray) -> jnp.ndarray:
        return elegy.nn.MultiHeadAttention(
            head_size=self.size,
            num_heads=self.num_heads,
            dropout=self.dropout,
        )(query=query, key=key)

    def self_attn(self, x: jnp.ndarray):
        return elegy.nn.MultiHeadAttention(
            head_size=self.size,
            num_heads=self.num_heads,
            dropout=self.dropout,
        )(x)

    def mlp(self, x: jnp.ndarray) -> jnp.ndarray:
        return FeedForward(dropout=self.dropout)(x)

    def norm(self, x: jnp.ndarray) -> jnp.ndarray:
        return elegy.nn.LayerNormalization()(x)


class FeedForward(elegy.Module):
    def __init__(self, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.mult = mult
        self.dropout = dropout

    def call(self, x):
        features = x.shape[-1]
        x = elegy.nn.Linear(features * self.mult)(x)
        x = jax.nn.gelu(x)
        x = elegy.nn.Dropout(self.dropout)(x)
        x = elegy.nn.Linear(features)(x)
        return x


class FourierFeatureEncoding(elegy.Module):
    def __init__(self, max_freq, num_bands):
        super().__init__()

        self.max_freq = max_freq
        self.num_bands = num_bands

    def call(self, input_tensor):
        batch, *axis, _ = input_tensor.shape

        position = jnp.stack(
            jnp.meshgrid(*[
                jnp.linspace(-1, 1, size) for size in axis
            ], indexing='ij'), 
            axis=-1
        )[..., None]

        scales = jnp.logspace(
            0.0, jnp.log2(self.max_freq / 2), num=self.num_bands, base=2
        )

        coef = position * scales * jnp.pi
        encoded_position = jax.vmap(
            lambda x: jnp.concatenate([jnp.sin(x), jnp.cos(x)], axis=-1)
        )(coef)
        encoded_position = jnp.concatenate([encoded_position, position], axis=-1)

        encoded_position = einops.rearrange(
            encoded_position, '... dims bands -> ... (dims bands)'
        )
        encoded_position = einops.repeat(
            encoded_position, '... -> batch ...', batch = batch
        )

        return jnp.concatenate((input_tensor, encoded_position), axis=-1)

def main(
    debug: bool = False,
    eager: bool = False,
    logdir: str = "runs",
    steps_per_epoch: int = 200,
    batch_size: int = 64,
    epochs: int = 100,
    size: int = 32,
    num_layers: int = 2,
    reps_per_layer: int = 3,
    num_heads: int = 8,
    dropout: float = 0.0,
    n_latents: int = 128,
    output_dim: int = 10,
):

    if debug:
        import debugpy

        print("Waiting for debugger...")
        debugpy.listen(5678)
        debugpy.wait_for_client()

    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    logdir = os.path.join(logdir, current_time)

    X_train, y_train, X_test, y_test = dataget.image.mnist(global_cache=True).get()

    print("X_train:", X_train.shape, X_train.dtype)
    print("y_train:", y_train.shape, y_train.dtype)
    print("X_test:", X_test.shape, X_test.dtype)
    print("y_test:", y_test.shape, y_test.dtype)

    model = elegy.Model(
        module=Perceiver(
            size=size,
            num_layers=num_layers,
            reps_per_layer=reps_per_layer,
            num_heads=num_heads,
            dropout=dropout,
            n_latents=n_latents,
            output_dim=output_dim,
        ),
        loss=[
            elegy.losses.SparseCategoricalCrossentropy(from_logits=True),
            # elegy.regularizers.GlobalL2(l=1e-4),
        ],
        metrics=elegy.metrics.SparseCategoricxalAccuracy(),
        optimizer=optax.adamw(3e-5),
        run_eagerly=eager,
    )

    model.init(X_train[:64], y_train[:64])

    model.summary(X_train[:64])

    history = model.fit(
        x=X_train,
        y=y_train,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        shuffle=True,
        callbacks=[elegy.callbacks.TensorBoard(logdir=logdir)],
    )

    elegy.utils.plot_history(history)

    # get random samples
    idxs = np.random.randint(0, 10000, size=(9,))
    x_sample = X_test[idxs]

    # get predictions
    y_pred = model.predict(x=x_sample)

    # plot and save results
    with SummaryWriter(os.path.join(logdir, "val")) as tbwriter:
        figure = plt.figure(figsize=(12, 12))
        for i in range(3):
            for j in range(3):
                k = 3 * i + j
                plt.subplot(3, 3, k + 1)
                plt.title(f"{np.argmax(y_pred[k])}")
                plt.imshow(x_sample[k], cmap="gray")
        # tbwriter.add_figure("Predictions", figure, 100)

    plt.show()

    print(
        "\n\n\nMetrics and images can be explored using tensorboard using:",
        f"\n \t\t\t tensorboard --logdir {logdir}",
    )


if __name__ == "__main__":
    typer.run(main)
