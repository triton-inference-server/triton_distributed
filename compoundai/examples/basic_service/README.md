from this dir:

```
compoundai serve basic:Frontend
```

to see the built manifest:

```
compoundai build -f bentofile.yaml

# the archive will be nested in ~/bentoml/bentos/basic/{bento_tag}/bento.yaml
```

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