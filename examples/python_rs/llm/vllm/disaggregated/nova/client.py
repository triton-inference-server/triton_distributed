import vllm
from vllm.utils import FlexibleArgumentParser
from vllm.engine.arg_utils import AsyncEngineArgs

# from nova_init.decorators import nova_endpoint, nova_service, nova_depends, nova_api
from compoundai import depends, nova_endpoint, service, api

from decode import Decode

@service(
    nova={
        "enabled": True,
        "namespace": "triton-init",
    }
)
class Client:
    # the original code -> points toward decode worker so we do that as well here
    decode = depends(Decode)

    @api()
    async def cmpl(self, prompt: str, max_tokens: int, temperature: float):
        stream = await self.decode.generate(
            {
                "prompt": prompt,
                "sampling_params": {
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
            }
        )
        async for response in stream:
            yield response.outputs[0].text
