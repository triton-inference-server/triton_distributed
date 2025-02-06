# triton-distributed

## Overview

This repository contains the core components of a distributed inference framework written in Rust. 

The core is creating graphs of AsyncEngines that can be pipelined together to form an inference graph. 

Component discovery and registration is managed over [etcd](https://etcd.io/). Component communication is managed over [NATS](https://nats.io/).


## Install Rust:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

## Rust Build

```bash
cargo build
```

## Run Tests

```bash
cargo test
```


