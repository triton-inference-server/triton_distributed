from kv_router_cai.processor import Processor

from compoundai import api
from compoundai.sdk.dependency import depends
from compoundai.sdk.service import service


@service(resources={"cpu": "1"}, traffic={"timeout": 60})  # Regular HTTP API
class Frontend:
    middle = depends(Processor)

    def __init__(self) -> None:
        print("Starting frontend")

    @api
    async def generate(self, text):
        """Stream results from the pipeline."""
        print(f"Frontend received: {text}")
        print(f"Frontend received type: {type(text)}")
        txt = RequestType(text=text)
        print(f"Frontend sending: {type(txt)}")
        async for response in self.middle.generate(txt.model_dump_json()):
            yield f"Frontend: {response}"
