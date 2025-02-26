import inspect
from typing import Annotated

from kv_router_cai.processor import Processor

from compoundai import api, depends, service


@service(
    traffic={"timeout": 10000},
    # image=NOVA_IMAGE
)
class Client:
    # the original code -> points to processor so we do that as well here
    processor = depends(Processor)

    def __init__(self):
        print("client init")

    # @api
    # async def cmpl(self, text: str):
    #     print(f"Frontend received: {text}")
    #     async for response in self.processor.generate(text):
    #         yield f"Frontend: {response}"

    @api
    async def cmpl(self, msg: str):
        # Call the generate method
        result = self.processor.generate(
            {
                "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                "messages": [{"role": "user", "content": msg}],
                "stream": True,
            }
        )

        # Check if the result is a coroutine or an async generator
        if inspect.iscoroutine(result):
            print("result is a coroutine")
            # If it's a coroutine, await it to get the async generator
            generator = await result
        else:
            # If it's already an async generator, use it directly
            print("result is an async generator")
            generator = result

        # Now iterate over the async generator
        async for response in generator:
            # Get the actual value from the Annotated object
            if isinstance(response, type(Annotated)):
                # Access the data as an attribute, not a method
                response_data = getattr(response, "data", response)
            else:
                response_data = response

            print("client response_data:", response_data)

            # Yield the actual data value
            yield response_data
