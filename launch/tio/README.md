# triton-llm service runner

`tio` is a tool for exploring the triton-distributed and triton-llm components.

## Install and start pre-requisites

Rust:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

## Build

- CUDA:

`cargo build --release --features mistralrs,cuda`

- MAC w/ Metal:

`cargo build --release --features mistralrs,metal`

- CPU only:

`cargo build --release --features mistralrs`

## Download a model from Hugging Face

For example one of these should be fast and good quality on almost any machine: https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF

## Run

*Text interface*

`./target/release/tio Llama-3.2-1B-Instruct-Q4_K_M.gguf` or path to a Hugging Face repo checkout instead of the GGUF.

*HTTP interface*

`./target/release/tio in=http --model-path Llama-3.2-1B-Instruct-Q4_K_M.gguf`

List the models: `curl localhost:8080/v1/models`

Send a request:
```
curl -d '{"model": "Llama-3.2-1B-Instruct-Q4_K_M", "max_tokens": 2049, "messages":[{"role":"user", "content": "What is the capital of South Africa?" }]}' -H 'Content-Type: application/json' http://localhost:8080/v1/chat/completions
```

*Multi-node*

Node 1:
```
tio in=http out=tdr://ns/backend/mistralrs
```

Node 2:
```
tio in=tdr://ns/backend/mistralrs out=mistralrs ~/llm_models/Llama-3.2-3B-Instruct
```

This will use etcd to auto-discover the model and NATS to talk to it. You can run multiple workers on the same endpoint and it will pick one at random each time.

The `ns/backend/mistralrs` are purely symbolic, pick anything as long as it has three parts, and it matches the other node.

Run `tio --help` for more options.

## sglang

```
uv venv
source .venv/bin/activate
uv pip install pip
uv pip install sgl-kernel --force-reinstall --no-deps
uv pip install "sglang[all]==0.4.2" --find-links https://flashinfer.ai/whl/cu124/torch2.4/flashinfer/
```
