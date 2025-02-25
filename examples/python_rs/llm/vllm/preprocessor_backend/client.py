import asyncio

import uvloop
from triton_distributed_rs import DistributedRuntime, triton_worker

uvloop.install()


@triton_worker()
async def worker(runtime: DistributedRuntime):
    """
    Instantiate a `backend` client and call the `generate` endpoint
    """
    # get endpoint
    endpoint = (
        runtime.namespace("triton-init").component("preprocessor").endpoint("generate")
    )

    # create client
    client = await endpoint.client()

    chat_completion_request = dict(
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        messages=[{"role": "user", "content": "what is deep learning?"}],
        max_tokens=64,
        stream=True,
    )

    # issue request
    stream = await client.generate(chat_completion_request)

    # process response
    async for resp in stream:
        print(resp)


asyncio.run(worker())
