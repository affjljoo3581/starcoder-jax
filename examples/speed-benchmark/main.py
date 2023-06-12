from __future__ import annotations

import argparse
import time

import jax
import numpy as np
from jax.experimental import mesh_utils
from jax.sharding import Mesh

from generation import Generator


def main(args: argparse.Namespace):
    rng = jax.random.PRNGKey(np.random.randint(100000))

    mesh = mesh_utils.create_device_mesh((args.data_parallel, args.model_parallel))
    mesh = Mesh(mesh, ("dp", "mp"))
    print(mesh)

    generator = Generator.from_huggingface(args.model, use_auth_token=True)
    generator.shard(mesh)
    print(f"[*] complete loading [{args.model}]")

    initial, hparams = generator.prepare_context(
        "import os\n",
        max_length=args.max_total_length,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    with mesh:
        tokens, rng = generator.generate_at_once(
            **initial, **hparams, rng=rng, max_new_tokens=args.max_new_tokens
        )
        tokens.block_until_ready()
    print("[*] finish compiling the generation code.")

    with mesh:
        timestamp = time.time()
        tokens, rng = generator.generate_at_once(
            **initial, **hparams, rng=rng, max_new_tokens=args.max_new_tokens
        )
        tokens = tokens.block_until_ready()
        time_per_token = (time.time() - timestamp) / (tokens != -1).sum()

        print("[*] generated sample:")
        print(generator.tokenizer.decode(tokens[0][tokens[0] != -1]))
        print(f"[*] generation speed: {time_per_token} sec/token")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="bigcode/starcoderbase")
    parser.add_argument("--data-parallel", type=int, default=1)
    parser.add_argument("--model-parallel", type=int, default=8)
    parser.add_argument("--max-total-length", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.92)
    main(parser.parse_args())
