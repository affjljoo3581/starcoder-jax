# technical-assistant

## Introduction
This is an example project to perform StarCoder as a technical assistant. The [technical-assistant prompt](https://huggingface.co/datasets/bigcode/ta-prompt) is one of the official StarCoder examples and this project is to reproduce in TPU environment with Jax/Flax. You can simply run by:
```bash
$ PYTHONPATH=../../src python3 main.py
```

## Example results

#### Human: How can I write a Python function to generate the nth Fibonacci number?
Assistant:  Sure.  Here’s one way.  Note that this is not a particularly efficient way to do it.

```python
def fibonacci(n):
   if n == 0:
        return 0
   elif n == 1:
        return 1
   else :
        return fibonacci(n-1) + fibonacci(n-2)
```

#### Human: How do I get the current date using shell commands? Explain how it works.
Assistant: 
```bash
date +"%d/%m/%Y"
```

#### Human: What's the meaning of life?
Assistant: 42

#### Human: Write a function in Javascript to reverse words in a given string.
Assistant: 
```javascript
function reverseWords(str) {
  return str.split(" ").reverse().join(" ");
}
```

#### Human: How to decode JSON into a typescript object
Assistant: 
```json
{
  "key": "value",
  "nested": {
    "key": "value"
  }
}
```

```ts
interface MyInterface {
  key: string;
  nested: {
    key: string;
  };
}
```

```ts
const myInterface: MyInterface = JSON.parse(
  '{ "key": "value", "nested": { "key": "value" }}',
);
```