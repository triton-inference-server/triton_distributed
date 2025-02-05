# Triton Distributed Python Bindings

Python bindings for the Triton distributed runtime system, enabling distributed computing capabilities for machine learning workloads.

## 🚀 Quick Start

1. Install `uv`: https://docs.astral.sh/uv/#getting-started
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Install `protoc` protobuf compiler: https://grpc.io/docs/protoc-installation/.

For example on an Ubuntu/Debian system:
```
apt install protobuf-compiler
```

3. Setup a virtualenv
```
cd python-wheels/triton-distributed
uv venv
source .venv/bin/activate
uv pip install maturin
```

4. Build and install triton_distributed wheel
```
maturin develop --uv
```

# Run Examples

## Pre-requisite

### Docker Compose

The simplest way to deploy the pre-requisite services is using
[docker-compose](https://docs.docker.com/compose/install/linux/),
defined in the project's root [docker-compose.yml](docker-compose.yml).

```
docker-compose up -d
```


This will deploy a [NATS.io](https://nats.io/) server and an [etcd](https://etcd.io/)
server used to communicate between and discover components at runtime.


### Local

To deploy the pre-requisite services locally instead of using `docker-compose`
above, you can manually launch each:

- [NATS.io](https://docs.nats.io/running-a-nats-service/introduction/installation) server with [Jetstream](https://docs.nats.io/nats-concepts/jetstream)
    - example: `nats-server -js --trace`
- `etcd` server
    - follow instructions in [etcd installation](https://etcd.io/docs/v3.5/install/) to start an `etcd-server` locally


## Hello World Example

1. Start 3 separate shells, and activate the virtual environment in each
```
cd python-wheels/triton-distributed
source .venv/bin/activate
```

2. In one shell (shell 1), run example server the instance-1
```
python3 ./examples/hello_world/server.py
```

3. (Optional) In another shell (shell 2), run example the server instance-2
```
python3 ./examples/hello_world/server.py
```

4. In the last shell (shell 3), run the example client:
```
python3 ./examples/hello_world/client.py
```

If you run the example client in rapid succession, and you started more than
one server instance above, you should see the requests from the client being
distributed across the server instances in each server's output. If only one
server instance is started, you should see the requests go to that server
each time.
