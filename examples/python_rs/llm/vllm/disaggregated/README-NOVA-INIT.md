Intructions on how to reproduce 1P1D VLLM disaggregated inference with Nova Init

Go to the root of the repository and run:

```bash
./container/build.sh --framework VLLM`
```

Enter the container and mount your workspace:

```bash
./container/run.sh -it --framework VLLM --mount-workspace
```

Start the venv

```bash
source /opt/triton/venv/bin/activate
```

Install the compoundai package

```bash
cd compoundai
uv pip install -e .
cd ..
```

Run the example

```bash
cd examples/python_rs/llm/vllm
compoundai serve disaggregated.client:Client
```

I would then port forward 3000 so you can view the Swagger UI and make requests. Or you can use:

```bash
curl -X POST \
 'http://localhost:3000/cmpl' \
 -H 'accept: text/event-stream' \
 -H 'Content-Type: application/json' \
 -d '{
"msg": "triton"
}'

```

It is also possible to do TP>1 via setting Cuda visible devices to more than 1 in the code for `decode.py` and `prefill.py`. We have not tested xPyD yet.

# Creating a container with the code.

This will get MUCH easier and work is already being done on compoundai in order to make this much simpler. In this branch, you will see a Dockerfile.cai that is a custom build of the triton-distributed container with the compoundai package installed. This just makes it easier to run the example.

First build the container

```bash
DOCKER_BUILDKIT=1 docker build -f Dockerfile.cai -t triton-distributed:cai .
```

From the same directory that we were in earlier, `cd examples/python_rs/llm/vllm`, run the following command to build the container. Note this is the same as the serve command but instead its build with containerize

```bash
compoundai build disaggregated.client:Client --containerize
```

Run it with

```bash
docker run --gpus all --network=host -u root --entrypoint /bin/bash client:44ay4rxsjklcgwzu -c "source /opt/triton/venv/bin/activate && compoundai serve"
```
