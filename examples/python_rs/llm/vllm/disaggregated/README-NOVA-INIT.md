Instructions on how to reproduce 1P1D VLLM disaggregated inference with Nova Init

1. `./container/build.sh --framework VLLM`
2. `./container/run.sh -it --framework VLLM --mount-workspace`
3. `source /opt/triton/venv/bin/activate`
4. `cd compoundai`
5. `uv pip install -e .`
6. `cd ..`
7. `cd examples/python_rs/llm/vllm`
8. `compoundai serve disaggregated.client:Client`


You can then visit localhost:3000 to make requests. Or you can run 

```
curl -X POST \
  'http://localhost:3000/cmpl' \
  -H 'accept: text/event-stream' \
  -H 'Content-Type: application/json' \
  -d '{
    "msg": "string"
  }'
```

In the client we create the OAI ChatCompletionRequest object and pass it to the disaggregated decode endpoint.

It is also possible to do TP>1 via setting Cuda visible devices to more than 1

Untested: xPyD - todo
