from this dir:

## Prerequisites
- Install triton-dist in your venv
- Before building or pushing, be sure to copy a wheel of triton_distributed_rs to the dist folder

## simple serve
```
compoundai serve basic:Frontend
```

## distributed serve
to run only standalone service and pass other running service as a dep
```
compoundai start --service-name Frontend --port 5001 --depends Middle=nova://inference/Middle

compoundai start --service-name Middle --depends Backend=nova://inference/Backend

compoundai start --service-name Backend
```

## building manifest
```
compoundai build -f bentofile.yaml

# the archive will be nested in ~/bentoml/bentos/basic/{bento_tag}/bento.yaml
```

## containerizing
containerizing:
- assumes you already created venv `uv venv`
- assumes you installed cai in that venv `uv pip install -e . # from compoundai dir`

instructions to setup isolated network, nats, etcd:
```
docker network create cai-network

# Run NATS (no port forwarding)
docker run -d --name nats-bento \
  --network cai-network \
  nats -js --trace

# Run etcd (no port forwarding)
docker run -d --name etcd-bento \
  --network cai-network \
  -e ALLOW_NONE_AUTHENTICATION=yes \
  bitnami/etcd
```

```
tdd

# build cai wheel
cd compoundai
uv build --wheel . --out-dir ./examples/basic_service/dist


# build container
cd examples/basic_service
# if you sourced the venv, you can just run the following
compoundai build --containerize -f bentofile.yaml
# else
# uv run compoundai build --containerize -f bentofile.yaml

# try to delete container in case it was left from previous runs
docker rm -f nova

docker run -it --name nova \
  --network cai-network \
  -p 5005:5005 \
  -e VIRTUAL_ENV=/opt/triton/venv \
  -e NATS_SERVER=nats://nats-bento:4222 \
  -e ETCD_ENDPOINTS=http://etcd-bento:2379 \
  --entrypoint=/bin/bash \
  docker.io/library/frontend:fwzyv3xqcslcgwzu \
  -c "uv run compoundai serve --port 5005"
```

## Start command
distributed serve
```
docker run -it --name nova \
  --network cai-network \
  -p 5005:5005 \
  -e NATS_SERVER=nats://nats-bento:4222 \
  -e ETCD_ENDPOINTS=http://etcd-bento:2379 \
  --entrypoint=sh \
  docker.io/library/basic:3jtxvvhtxolcgwzu \
  -c "uv run compoundai start --service-name Frontend --port 5005 --depends Middle=nova://inference/Middle"

docker run -it --name nova-2 \
  --network cai-network \
  -e NATS_SERVER=nats://nats-bento:4222 \
  -e ETCD_ENDPOINTS=http://etcd-bento:2379 \
  --entrypoint=sh \
  docker.io/library/basic:3jtxvvhtxolcgwzu \
  -c "uv run compoundai start --service-name Middle --depends Backend=nova://inference/Backend"

docker run -it --name nova-3 \
  --network cai-network \
  -e NATS_SERVER=nats://nats-bento:4222 \
  -e ETCD_ENDPOINTS=http://etcd-bento:2379 \
  --entrypoint=sh \
  docker.io/library/basic:3jtxvvhtxolcgwzu \
  -c "uv run compoundai start --service-name Backend"
```

single serve
```
docker run -it --name nova \
  --network cai-network \
  -p 5005:5005 \
  -e NATS_SERVER=nats://nats-bento:4222 \
  -e ETCD_ENDPOINTS=http://etcd-bento:2379 \
  --entrypoint=sh \
  docker.io/library/basic:3jtxvvhtxolcgwzu \
  -c "uv run compoundai serve --port 5005"
```
