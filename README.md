# smolcde

A small library for conditional density estimation using masked
autoregressive flows, with Python & C implementations and a small
CLI.

## Setup

The CLI/C implementation, is usable from source via clone repo & `cmake . && make`, otherwise download from [releases on GitHub](https://github.com/maedoc/smolcde/releases), where binaries for each platform are available.

The Python implementation depends on numpy, scipy and autograd only, making it easy to just drop the file `cde.py` into your project. 

You can run the tests with Python
```
uv venv env; . env/bin/activate
uv pip install pytest scipy autograd scikit_learn tqdm
pytest
```

## Usage examples

Please see [README.ipynb](README.ipynb) for info on usage & examples.
