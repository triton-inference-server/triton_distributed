import vllm
from vllm.utils import FlexibleArgumentParser
from vllm.engine.arg_utils import AsyncEngineArgs
import json
from typing import Annotated

# from nova_init.decorators import nova_endpoint, nova_service, nova_depends, nova_api
from compoundai import depends, nova_endpoint, service, api

from disaggregated.decode import Decode

@service()
class Client:
    # the original code -> points toward decode worker so we do that as well here
    decode = depends(Decode)

    def __init__(self):
        print("client init")

    @api
    async def cmpl(self, msg: str):
        
        # Don't await the generator - directly async iterate over it
        decgen = self.decode.generate(
            {
                "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                "messages": [
                    {
                        "role": "user",
                        "content": msg
                    }
                ],
                "stream": True
            }
        )
        async for response in decgen:
            # Get the actual value from the Annotated object
            if isinstance(response, type(Annotated)):
                # Access the data as an attribute, not a method
                response_data = getattr(response, 'data', response)
            else:
                response_data = response
                
            print("client response_data:", response_data)
            
            # Yield the actual data value
            yield response_data
