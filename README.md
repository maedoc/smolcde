# smolcde

A small library for conditional density estimation using masked
autoregressive flows, with Python & C implementations and a small
CLI.

build the cli 
```
cmake . && make
```

run the tests
```
uv venv env; . env/bin/activate
uv pip install pytest scipy autograd scikit_learn tqdm
pytest
```

clean up
```
git clean -xfd
```
