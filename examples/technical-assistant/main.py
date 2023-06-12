from __future__ import annotations

import argparse

import jax
import numpy as np
from jax.experimental import mesh_utils
from jax.sharding import Mesh

from generation import Generator

TA_PROMPT_SEPARATOR = "-----"


def main(args: argparse.Namespace):
    rng = jax.random.PRNGKey(np.random.randint(100000))

    mesh = mesh_utils.create_device_mesh((args.data_parallel, args.model_parallel))
    mesh = Mesh(mesh, ("dp", "mp"))
    print(mesh)

    generator = Generator.from_huggingface(args.model, use_auth_token=True)
    generator.shard(mesh)
    print(f"[*] complete loading [{args.model}]")

    with open("TA_prompt_v1.txt") as fp:
        base_prompt = fp.read()

    while True:
        prompt = base_prompt
        prompt += "Human: " + input("Human: ") + "\n"
        prompt += "Assistant: "

        print("Assistant: ", end="")
        generated = ""

        with mesh:
            initial, hparams = generator.prepare_context(
                prompt,
                max_length=args.max_total_length,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            outputs = generator.generate_from_context(**initial, **hparams, rng=rng)
            token = generator.tokenizer.decode(int(outputs[0][0]))
            generated += token
            print(token, end="", flush=True)

            for _ in range(args.max_new_tokens):
                outputs = generator.generate_next_tokens(*outputs, **hparams)
                token = generator.tokenizer.decode(int(outputs[0][0]))
                generated += token

                if generated.endswith("\nHuman"):
                    break
                if generated.endswith(f"\n{TA_PROMPT_SEPARATOR}"):
                    break
                print(token, end="", flush=True)
        rng = outputs[2]


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
