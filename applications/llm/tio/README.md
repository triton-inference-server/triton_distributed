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

Run `tio --help` for more options.

