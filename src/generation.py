from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

import chex
import jax
import jax.numpy as jnp
from jax.experimental.pjit import pjit
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from miscellaneous import convert_weights, get_conversion_rules, get_sharding_rules
from modeling import Transformer


def top_p_sampling(
    logits: chex.Array, rng: chex.PRNGKey, top_p: chex.Array
) -> chex.Array:
    sorted_logits, sorted_indices = jax.lax.top_k(logits, logits.shape[1])

    mask = jnp.cumsum(jax.nn.softmax(sorted_logits, axis=1), axis=1) < top_p
    mask = jnp.roll(mask, 1, axis=1).at[:, 0].set(True)

    sorted_logits = sorted_logits - 1e9 * (1 - mask.astype(sorted_logits.dtype))
    indices = jax.random.categorical(rng, sorted_logits)
    return jnp.take_along_axis(sorted_indices, indices[:, None], axis=1)[:, 0]


@partial(pjit, static_argnums=(0,))
def _generate_from_context(
    model,
    variables: chex.ArrayTree,
    tokens: chex.Array,
    mask: chex.Array,
    rng: chex.PRNGKey,
    temperature: chex.Array,
    top_p: chex.Array,
) -> tuple[chex.Array, chex.ArrayTree, chex.PRNGKey]:
    tokens = jax.lax.with_sharding_constraint(tokens, PartitionSpec("dp", None))
    mask = jax.lax.with_sharding_constraint(mask, PartitionSpec("dp", None))

    logits, cache = model.apply(variables, tokens, mask, mutable=["cache"])
    logits = logits[:, -1, :].astype(jnp.float32) / temperature

    rng, new_rng = jax.random.split(rng)
    new_tokens = top_p_sampling(logits / temperature, rng, top_p)
    return new_tokens, cache["cache"], new_rng


@partial(pjit, static_argnums=(0,))
def _generate_next_tokens(
    model,
    variables: chex.ArrayTree,
    new_tokens: chex.Array,
    rng: chex.PRNGKey,
    temperature: chex.Array,
    top_p: chex.Array,
) -> tuple[chex.Array, chex.ArrayTree, chex.PRNGKey]:
    tokens = jax.lax.with_sharding_constraint(new_tokens, PartitionSpec("dp"))
    logits, cache = model.apply(variables, tokens[:, None], mutable=["cache"])
    logits = logits[:, -1, :].astype(jnp.float32) / temperature

    rng, new_rng = jax.random.split(rng)
    new_tokens = top_p_sampling(logits / temperature, rng, top_p)
    return new_tokens, cache["cache"], new_rng


@partial(pjit, static_argnums=(0, 7, 8))
def _generate_at_once(
    model,
    variables: chex.ArrayTree,
    tokens: chex.Array,
    mask: chex.Array,
    rng: chex.PRNGKey,
    temperature: chex.Array,
    top_p: chex.Array,
    max_new_tokens: int = 1024,
    eos_token_id: int = 1,
) -> tuple[chex.Array, chex.PRNGKey]:
    outputs = _generate_from_context(
        model, variables, tokens, mask, rng, temperature, top_p
    )
    generated = jnp.full((tokens.shape[0], max_new_tokens), -1, dtype=jnp.int32)
    generated = jnp.roll(generated, -1, 1).at[:, -1].set(outputs[0])

    def cond_fn(state: chex.ArrayTree) -> bool:
        not_full = (state[3] == -1).any(1).all()
        not_ended = (state[3] != eos_token_id).all(1).any()
        return not_full & not_ended

    def body_fn(state: chex.ArrayTree) -> chex.ArrayTree:
        new_tokens, cache, rng, generated = state
        outputs = _generate_next_tokens(
            model, {**variables, "cache": cache}, new_tokens, rng, temperature, top_p
        )
        update = jnp.where((generated == eos_token_id).any(1), -1, outputs[0])
        return *outputs, jnp.roll(generated, -1, 1).at[:, -1].set(update)

    results = jax.lax.while_loop(cond_fn, body_fn, init_val=(*outputs, generated))
    return results[3], results[2]


@dataclass
class Generator:
    model: Transformer
    params: chex.ArrayTree
    tokenizer: PreTrainedTokenizerBase

    def to_token(self, token_id: int) -> str:
        return self.tokenizer.decode(token_id)

    def shard(self, mesh: Mesh):
        partition_spec = get_sharding_rules(self.model)
        partition_spec = jax.tree_map(partial(NamedSharding, mesh), partition_spec)
        self.params = jax.tree_map(jax.device_put, self.params, partition_spec)

    def generate_from_context(
        self,
        tokens: chex.Array,
        mask: chex.Array,
        rng: chex.PRNGKey,
        temperature: chex.Array,
        top_p: chex.Array,
    ) -> tuple[chex.Array, chex.ArrayTree, chex.PRNGKey]:
        # We wrap the arrays with `jnp.asarray` to restrict the input arrays and prevent
        # recompiling the function.
        return _generate_from_context(
            self.model,
            {"params": self.params},
            jnp.asarray(tokens, dtype=jnp.int32),
            jnp.asarray(mask, dtype=jnp.bool_),
            rng,
            jnp.asarray(temperature, dtype=jnp.float32),
            jnp.asarray(top_p, dtype=jnp.float32),
        )

    def generate_next_tokens(
        self,
        new_tokens: chex.Array,
        cache: chex.ArrayTree,
        rng: chex.PRNGKey,
        temperature: chex.Array,
        top_p: chex.Array,
    ) -> tuple[chex.Array, chex.ArrayTree, chex.PRNGKey]:
        # We wrap the arrays with `jnp.asarray` to restrict the input arrays and prevent
        # recompiling the function.
        return _generate_next_tokens(
            self.model,
            {"params": self.params, "cache": cache},
            jnp.asarray(new_tokens, dtype=jnp.int32),
            rng,
            jnp.asarray(temperature, dtype=jnp.float32),
            jnp.asarray(top_p, dtype=jnp.float32),
        )

    def generate_at_once(
        self,
        tokens: chex.Array,
        mask: chex.Array,
        rng: chex.PRNGKey,
        temperature: chex.Array,
        top_p: chex.Array,
        max_new_tokens: int = 1024,
    ) -> tuple[chex.Array, chex.PRNGKey]:
        # We wrap the arrays with `jnp.asarray` to restrict the input arrays and prevent
        # recompiling the function.
        return _generate_at_once(
            self.model,
            {"params": self.params},
            jnp.asarray(tokens, dtype=jnp.int32),
            jnp.asarray(mask, dtype=jnp.bool_),
            rng,
            jnp.asarray(temperature, dtype=jnp.float32),
            jnp.asarray(top_p, dtype=jnp.float32),
            max_new_tokens,
            self.tokenizer.eos_token_id,
        )

    def prepare_context(
        self,
        text: str,
        max_length: int = 8192,
        temperature: float = 0.8,
        top_p: float = 0.92,
    ) -> tuple[dict[str, chex.Array], dict[str, chex.Array]]:
        encodings = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="np",
        )
        initial_inputs = {
            "tokens": jnp.asarray(encodings.input_ids, dtype=jnp.int32),
            "mask": jnp.asarray(encodings.attention_mask, dtype=jnp.bool_),
        }
        generation_params = {
            "temperature": jnp.asarray([temperature], dtype=jnp.float32),
            "top_p": jnp.asarray([top_p], dtype=jnp.float32),
        }
        return initial_inputs, generation_params

    @staticmethod
    def from_huggingface(name: str, **kwargs: Any) -> Generator:
        tokenizer = AutoTokenizer.from_pretrained(name, **kwargs)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        hf_model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
        model = Transformer(
            vocab_size=hf_model.config.vocab_size,
            max_length=hf_model.config.n_positions,
            layers=hf_model.config.n_layer,
            dim=hf_model.config.n_embd,
            heads=hf_model.config.n_head,
            hidden=hf_model.config.n_inner,
            eps=hf_model.config.layer_norm_epsilon,
        )
        params = convert_weights(hf_model.state_dict(), get_conversion_rules(model))
        return Generator(model, params, tokenizer)
