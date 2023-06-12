from __future__ import annotations

import asyncio
import json
import multiprocessing as mp
import os
import random
import shutil
from functools import partial

import jax
import numpy as np
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from prompting import DIALOGUE_SEPARATOR, create_input_prompt
from sse_starlette import EventSourceResponse
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from generation import Generator

HF_MODEL_NAME = os.environ.get("HF_MODEL_NAME", "bigcode/starcoderbase")
GIT_REPO_STORE_DIR = os.environ.get("GIT_REPO_STORE_DIR", "git-repo-store")
TPU_MESH_DATA_PARALLELISM = int(os.environ.get("TPU_MESH_DATA_PARALLELISM", "1"))
TPU_MESH_MODEL_PARALLELISM = int(os.environ.get("TPU_MESH_MODEL_PARALLELISM", "8"))

MAX_TOTAL_PROMPT_LENGTH = int(os.environ.get("MAX_TOTAL_PROMPT_LENGTH", "8192"))
MAX_README_TOKENS = int(os.environ.get("MAX_README_TOKENS", "1024"))


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tokenizer: PreTrainedTokenizerBase
input_queue: mp.Queue
output_queue: mp.Queue
exit_queue: mp.Queue
stream_queue_table: dict[str, asyncio.Queue] = {}


def generate_subscribe_id() -> str:
    return "".join(random.choices("0123456789abcdef", k=16))


def generate_worker_fn(
    input_queue: mp.Queue, output_queue: mp.Queue, exit_queue: mp.Queue
):
    rng = jax.random.PRNGKey(np.random.randint(100000))
    exited_sids = set()

    mesh_shape = (TPU_MESH_DATA_PARALLELISM, TPU_MESH_MODEL_PARALLELISM)
    mesh = Mesh(mesh_utils.create_device_mesh(mesh_shape), ("dp", "mp"))
    print(mesh)

    generator = Generator.from_huggingface(HF_MODEL_NAME, use_auth_token=True)
    generator.shard(mesh)
    print(f"[*] complete loading [{HF_MODEL_NAME}]")

    with mesh:
        while True:
            sid, prompt = input_queue.get()
            if sid in exited_sids:
                break

            initial, hparams = generator.prepare_context(
                prompt,
                max_length=MAX_TOTAL_PROMPT_LENGTH,
                temperature=0.8,
                top_p=0.92,
            )

            outputs = generator.generate_from_context(**initial, **hparams, rng=rng)
            output_queue.put((sid, generator.tokenizer.decode(int(outputs[0][0]))))

            for _ in range(MAX_README_TOKENS):
                while not exit_queue.empty():
                    exited_sids.add(exit_queue.get())
                if sid in exited_sids:
                    break

                outputs = generator.generate_next_tokens(*outputs, **hparams)
                output_queue.put((sid, generator.tokenizer.decode(int(outputs[0][0]))))

            output_queue.put((sid, None))


async def queue_observer_fn():
    loop = asyncio.get_event_loop()
    while True:
        sid, token = await loop.run_in_executor(None, output_queue.get)
        if sid in stream_queue_table:
            await stream_queue_table[sid].put(token)


@app.on_event("startup")
async def startup_event():
    global tokenizer, input_queue, output_queue, exit_queue
    shutil.rmtree(GIT_REPO_STORE_DIR, ignore_errors=True)

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME, use_auth_token=True)
    input_queue = mp.Queue()
    output_queue = mp.Queue()
    exit_queue = mp.Queue()

    mp.Process(
        target=generate_worker_fn,
        args=(input_queue, output_queue, exit_queue),
        daemon=True,
    ).start()

    asyncio.get_event_loop().create_task(queue_observer_fn())


@app.get("/generate")
async def generate(gitUrl: str, request: Request) -> Response:
    os.makedirs(GIT_REPO_STORE_DIR, exist_ok=True)
    os.system(f"cd {GIT_REPO_STORE_DIR}; git clone {gitUrl}")

    repo_dir = os.path.basename(gitUrl).replace(".git", "")
    repo_dir = os.path.join(GIT_REPO_STORE_DIR, repo_dir)

    prompt = create_input_prompt(
        repo_dir,
        tokenizer,
        gitUrl,
        max_length=MAX_TOTAL_PROMPT_LENGTH - MAX_README_TOKENS,
    )
    while (sid := generate_subscribe_id()) in stream_queue_table:
        pass

    stream_queue_table[sid] = asyncio.Queue()
    await asyncio.get_event_loop().run_in_executor(
        None, partial(input_queue.put, (sid, prompt))
    )

    async def stream_generator_from_queue():
        generated = ""
        while not generated.endswith(DIALOGUE_SEPARATOR):
            if await request.is_disconnected():
                break
            token = await stream_queue_table[sid].get()
            if token is None or token == tokenizer.eos_token:
                break
            generated += token
            yield json.dumps(token)

        asyncio.get_event_loop().run_in_executor(None, partial(exit_queue.put, sid))
        stream_queue_table.pop(sid)
        shutil.rmtree(repo_dir, ignore_errors=True)

    return EventSourceResponse(stream_generator_from_queue())
