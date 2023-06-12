# readme-generation

## Introduction
This is an example project of `starcoder-jax`. This example launches a fastapi backend server which serves a README.md document generation from GitHub repository URLs. You can see the detailed specification in [the below section](#api-documentation).

## Requirements
In addition to the original dependencies, this example requires the below libraries:
* fastapi
* sseclient-py
* uvicorn

## Usage
Using the below command in the terminal to launch the server:
```bash
$ PYTHONPATH=../../src uvicorn app:app --host localhost --port 8000
```

## API Documentation

### GET - http://127.0.0.1:8000/generate

#### Request:
```text
gitUrl=https://github.com/wandb/wandb
```

#### SSE Response:
```
"world\".\n"
```
The text from SSE is encoded by JSON format, thus it is required to decode by `JSON.parse()`. Note that the value contains a subword token only without any key-value pairs, so you can directly use the decoded text. Furthermore, because the tokens are encoded by JSON format, the special characters like line-break, spacing and quotes are also escaped and you can just aggregate the tokens without any joining separators.

Here is an example for python client:
```python
import json
import requests
import sseclient

text = ""
with requests.get("http://127.0.0.1:8000/generate", params={"gitUrl": url}, stream=True) as resp:
    for event in sseclient.SSEClient(resp).events():
        text += json.loads(event.data)
```