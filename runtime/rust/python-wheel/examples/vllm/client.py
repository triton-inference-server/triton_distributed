import asyncio
import uvloop

from vllm.utils import FlexibleArgumentParser

from protocol import Request
from triton_distributed_rs import DistributedRuntime, triton_worker


@triton_worker()
async def worker(runtime: DistributedRuntime, prompt: str):
    """
    Instantiate a `backend` client and call the `generate` endpoint
    """
    # get endpoint
    endpoint = (
        runtime.namespace("triton-init").component("vllm").endpoint("generate")
    )

    # create client
    client = await endpoint.client()

    # list the endpoints
    print(client.endpoint_ids())

    # issue request
    stream = await client.generate(Request(prompt="what is the capital of france?", sampling_params={"temperature": 0.5}).model_dump_json())

    # process response
    async for resp in stream:
        print(resp)


if __name__ == "__main__":
    uvloop.install()

    parser = FlexibleArgumentParser()
    parser.add_argument("--prompt", type=str, default="what is the capital of france?")
    args = parser.parse_args()

    asyncio.run(worker(args.prompt))
