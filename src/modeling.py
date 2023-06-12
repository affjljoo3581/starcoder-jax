from __future__ import annotations

import chex
import flax.linen as nn
import jax.numpy as jnp


class Attention(nn.Module):
    dim: int
    heads: int

    def setup(self):
        head_dim = self.dim // self.heads
        self.wq = nn.DenseGeneral((self.heads, head_dim), dtype=jnp.bfloat16)
        self.wk = nn.Dense(head_dim, dtype=jnp.bfloat16)
        self.wv = nn.Dense(head_dim, dtype=jnp.bfloat16)
        self.wo = nn.DenseGeneral(self.dim, axis=(-2, -1), dtype=jnp.bfloat16)

    def update_cache(self, name: str, x: chex.Array) -> chex.Array:
        if (cache := self.get_variable("cache", name)) is not None:
            x = jnp.roll(cache, -x.shape[1], axis=1).at[:, -x.shape[1] :].set(x)
        self.put_variable("cache", name, x)
        return x

    def __call__(self, x: chex.Array, attn_bias: chex.Array) -> chex.Array:
        q = self.wq(x)
        k = self.update_cache("k", self.wk(x))
        v = self.update_cache("v", self.wv(x))

        p = jnp.einsum("bqhd,bkd->bhqk", q, k) / k.shape[-1] ** 0.5
        x = jnp.einsum("bhqk,bkd->bqhd", nn.softmax(p + attn_bias, axis=3), v)
        return self.wo(x)


class FeedForward(nn.Module):
    dim: int
    hidden: int

    def setup(self):
        self.w1 = nn.Dense(self.hidden, dtype=jnp.bfloat16)
        self.w2 = nn.Dense(self.dim, dtype=jnp.bfloat16)

    def __call__(self, x: chex.Array) -> chex.Array:
        return self.w2(nn.gelu(self.w1(x), approximate=False))


class TransformerLayer(nn.Module):
    dim: int
    heads: int
    hidden: int
    eps: float = 1e-5

    def setup(self):
        self.attn = Attention(self.dim, self.heads)
        self.ff = FeedForward(self.dim, self.hidden)

        self.attn_norm = nn.LayerNorm(self.eps, dtype=jnp.bfloat16)
        self.ff_norm = nn.LayerNorm(self.eps, dtype=jnp.bfloat16)

    def __call__(self, x: chex.Array, attn_bias: chex.Array) -> chex.Array:
        x = x + self.attn(self.attn_norm(x), attn_bias)
        x = x + self.ff(self.ff_norm(x))
        return x


class Transformer(nn.Module):
    vocab_size: int
    max_length: int
    layers: int
    dim: int
    heads: int
    hidden: int
    eps: float = 1e-5

    def setup(self):
        self.wte = nn.Embed(self.vocab_size, self.dim, dtype=jnp.bfloat16)
        self.wpe = nn.Embed(self.max_length, self.dim, dtype=jnp.bfloat16)

        layer_args = (self.dim, self.heads, self.hidden, self.eps)
        self.layer = [TransformerLayer(*layer_args) for _ in range(self.layers)]

        self.head = nn.Dense(self.vocab_size, use_bias=False, dtype=jnp.bfloat16)
        self.head_norm = nn.LayerNorm(self.eps, dtype=jnp.bfloat16)

    def __call__(self, x: chex.Array, mask: chex.Array | None = None) -> chex.Array:
        if mask is None:
            mask = self.get_variable("cache", "mask")
            mask = jnp.roll(mask, -x.shape[1], axis=1).at[:, -x.shape[1] :].set(True)
        self.put_variable("cache", "mask", mask)

        # Create an attention bias to mask the attention probability which should be
        # ignored. To mask the future tokens, `jnp.tril` is used to the extended
        # attention bias array. We use `-1e9` which is relatively high penalty to make
        # the exponential value to zero.
        attn_bias = jnp.repeat(mask[:, None, None, :], x.shape[1], axis=2)
        attn_bias = jnp.tril(attn_bias, k=attn_bias.shape[3] - attn_bias.shape[2])
        attn_bias = -1e9 * (1 - attn_bias.astype(jnp.bfloat16))

        # Embed the tokens and their position. The index of each token can be inferred
        # by cumulatively summing the attention mask.
        p = jnp.cumsum(mask, axis=1, dtype=jnp.int32)[:, -x.shape[1] :] - 1
        x = self.wte(x) + self.wpe(p)

        for layer in self.layer:
            x = layer(x, attn_bias)
        return self.head(self.head_norm(x))
