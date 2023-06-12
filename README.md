# starcoder-jax

## Introduction

This repository is a Jax/Flax implementation of the [StarCoder](https://github.com/bigcode-project/starcoder) model. We implement the inference code of [GPTBigCode](https://huggingface.co/docs/transformers/model_doc/gpt_bigcode) architecture. With this repository, you can run GPTBigCode based models such as [starcoder](https://huggingface.co/bigcode/starcoder), [starcoderbase](https://huggingface.co/bigcode/starcoderbase) and [starcoderplus](https://huggingface.co/bigcode/starcoderplus).

The StarCoder models have 15.5B parameters and it requires about 63GB of memory for parameters only. Since tpu-v3-8 consists of 8 cores of 16GB, it is necessary to shard the parameters into multiple devices. Therefore this repository provides 2D parallelism (data parallelism and model parallelism) for inference.

## Requirements

The below libraries are required to run the starcoder.

- jax
- flax
- chex
- torch
- transformers

If you are trying to run on Cloud TPU VM, run the below commands:

```bash
$ pip install torch --index-url https://download.pytorch.org/whl/cpu
$ pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
$ pip install flax chex transformers
```
Also you may need to login to the huggingface hub. Use the below command:
```bash
$ ~/.local/bin/huggingface-cli login [token]
```
## Usage

This repository provides an interface to generate a text from the model. First of all, create a device mesh for parallelism and load model weights. The `Generator` class will automatically convert the PyTorch weights to Jax/Flax format. Note that you must specify your huggingface API token to load StarCoder models (because of the licence agreement).

```python
# Define a parallelism rule.
mesh = Mesh(mesh_utils.create_device_mesh((1, 8)), ("dp", "mp"))

# Load the model from huggingface and shard the parameters into multiple devices.
generator = Generator.from_huggingface("bigcode/starcoder", use_auth_token=True)
generator.shard(mesh)
```

After loading the weights, you should prepare an initial input for the prompt context. The `Generator` class also provides a method to encode the text and its generation options:
```python
context = """
def print_len(x):
    '''print the length of the string.'''
"""

initial, hparams = generator.prepare_context(
    context,
    max_length=8192,
    temperature=0.8,
    top_p=0.92,
)
```
The output `hparams` contains the hyperparameters for generation (`temperature` and `top_p`). As you can see below, it is reused while predicting next tokens. You can stack the hyperparameters with their initial inputs to make a batch with using different generation options.

### Iterative generation

Like ChatGPT, you can iteratively generate next tokens from the model for streaming the generation progress.

```python
with mesh:
    outputs = generator.generate_from_context(**initial, **hparams, rng=rng)
    print(generator.tokenizer.decode(int(outputs[0][0])), end="", flush=True)

    for _ in range(1024):
        outputs = generator.generate_next_tokens(*outputs, **hparams)
        print(generator.tokenizer.decode(int(outputs[0][0])), end="", flush=True)
```

### Generate at once

Instead, you can generate a sentence at once like Bard. It can be accomplished by putting the above codes in a single function and compiling it. `generator.generate_at_once` performs the above codes with aggregating the tokens.

```python
with mesh:
    tokens, rng = generator.generate_at_once(**initial, **hparams, rng=rng, max_new_tokens=1024)
print(generator.tokenizer.decode(tokens[0][tokens[0] != -1]))
```

For more details, check out the [examples](#examples).

## Examples

- [README.md generation](examples/readme-generation)
- [Technical Assistant](examples/technical-assistant)
- [Inference Speed Benchmark](examples/speed-benchmark)

## Acknowledgements
[Tensorflow Research Cloud](https://sites.research.google/trc/about/) provides the TPU Resources for testing.

## License
[MIT](./LICENSE)