from __future__ import annotations

from typing import Any, NamedTuple

import chex
import flax.traverse_util
import jax.numpy as jnp
import torch
from jax.sharding import PartitionSpec

from modeling import Transformer


class ConversionRule(NamedTuple):
    name: str
    transpose: bool = False
    slicing: tuple[slice, ...] | None = None
    unflatten: dict[str, Any] | None = None
    dtype: jnp.dtype = jnp.bfloat16


def convert_weights(
    state_dict: dict[str, torch.Tensor], rules: dict[str, ConversionRule]
) -> chex.ArrayTree:
    params = {}
    for name, rule in rules.items():
        param = state_dict[rule.name]
        if rule.transpose:
            param = param.transpose(0, 1)
        if rule.slicing is not None:
            param = param[rule.slicing]
        if rule.unflatten is not None:
            param = param.unflatten(**rule.unflatten)
        params[name] = param.numpy().astype(jnp.bfloat16)
    return flax.traverse_util.unflatten_dict(params, sep=".")


def get_conversion_rules(model: Transformer) -> dict[str, ConversionRule]:
    head_dim = model.dim // model.heads

    WEIGHT_CONVERSION_RULES = {
        "wte.embedding": ConversionRule("transformer.wte.weight"),
        "wpe.embedding": ConversionRule("transformer.wpe.weight"),
        "layer_{}.attn.wq.kernel": ConversionRule(
            "transformer.h.{}.attn.c_attn.weight",
            transpose=True,
            slicing=(slice(None), slice(-2 * head_dim)),
            unflatten=dict(dim=1, sizes=(model.heads, -1)),
        ),
        "layer_{}.attn.wk.kernel": ConversionRule(
            "transformer.h.{}.attn.c_attn.weight",
            transpose=True,
            slicing=(slice(None), slice(-2 * head_dim, -head_dim)),
        ),
        "layer_{}.attn.wv.kernel": ConversionRule(
            "transformer.h.{}.attn.c_attn.weight",
            transpose=True,
            slicing=(slice(None), slice(-1 * head_dim, None)),
        ),
        "layer_{}.attn.wo.kernel": ConversionRule(
            "transformer.h.{}.attn.c_proj.weight",
            transpose=True,
            unflatten=dict(dim=0, sizes=(model.heads, -1)),
        ),
        "layer_{}.attn.wq.bias": ConversionRule(
            "transformer.h.{}.attn.c_attn.bias",
            slicing=slice(-2 * head_dim),
            unflatten=dict(dim=0, sizes=(model.heads, -1)),
        ),
        "layer_{}.attn.wk.bias": ConversionRule(
            "transformer.h.{}.attn.c_attn.bias", slicing=slice(-2 * head_dim, -head_dim)
        ),
        "layer_{}.attn.wv.bias": ConversionRule(
            "transformer.h.{}.attn.c_attn.bias", slicing=slice(-head_dim, None)
        ),
        "layer_{}.attn.wo.bias": ConversionRule("transformer.h.{}.attn.c_proj.bias"),
        "layer_{}.ff.w1.kernel": ConversionRule(
            "transformer.h.{}.mlp.c_fc.weight", transpose=True
        ),
        "layer_{}.ff.w2.kernel": ConversionRule(
            "transformer.h.{}.mlp.c_proj.weight", transpose=True
        ),
        "layer_{}.ff.w1.bias": ConversionRule("transformer.h.{}.mlp.c_fc.bias"),
        "layer_{}.ff.w2.bias": ConversionRule("transformer.h.{}.mlp.c_proj.bias"),
        "layer_{}.attn_norm.scale": ConversionRule("transformer.h.{}.ln_1.weight"),
        "layer_{}.attn_norm.bias": ConversionRule("transformer.h.{}.ln_1.bias"),
        "layer_{}.ff_norm.scale": ConversionRule("transformer.h.{}.ln_2.weight"),
        "layer_{}.ff_norm.bias": ConversionRule("transformer.h.{}.ln_2.bias"),
        "head.kernel": ConversionRule("lm_head.weight", transpose=True),
        "head_norm.scale": ConversionRule("transformer.ln_f.weight"),
        "head_norm.bias": ConversionRule("transformer.ln_f.bias"),
    }

    conversion_rules = {}
    for k, v in WEIGHT_CONVERSION_RULES.items():
        for i in range(model.layers):
            conversion_rules[k.format(i)] = ConversionRule(v[0].format(i), *v[1:])
    return conversion_rules


def get_sharding_rules(model: Transformer) -> chex.ArrayTree:
    WEIGHT_SHARDING_RULES = {
        "wte.embedding": ("mp", None),
        "wpe.embedding": ("mp", None),
        "layer_{}.attn.wq.kernel": (None, "mp", None),
        "layer_{}.attn.wk.kernel": (None, None),
        "layer_{}.attn.wv.kernel": (None, None),
        "layer_{}.attn.wo.kernel": ("mp", None, None),
        "layer_{}.attn.wq.bias": ("mp", None),
        "layer_{}.attn.wk.bias": (None,),
        "layer_{}.attn.wv.bias": (None,),
        "layer_{}.attn.wo.bias": (None,),
        "layer_{}.ff.w1.kernel": (None, "mp"),
        "layer_{}.ff.w2.kernel": ("mp", None),
        "layer_{}.ff.w1.bias": ("mp",),
        "layer_{}.ff.w2.bias": (None,),
        "layer_{}.attn_norm.scale": (None,),
        "layer_{}.attn_norm.bias": (None,),
        "layer_{}.ff_norm.scale": (None,),
        "layer_{}.ff_norm.bias": (None,),
        "head.kernel": (None, "mp"),
        "head_norm.scale": (None,),
        "head_norm.bias": (None,),
    }

    sharding_rules = {}
    for k, v in WEIGHT_SHARDING_RULES.items():
        for i in range(model.layers):
            sharding_rules[k.format(i)] = PartitionSpec(*v)
    return flax.traverse_util.unflatten_dict(sharding_rules, sep=".")
