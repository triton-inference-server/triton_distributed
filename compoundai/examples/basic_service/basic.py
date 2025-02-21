from compoundai.sdk.service import service
from compoundai.sdk.decorators import nova_endpoint
from compoundai.sdk.dependency import depends
from compoundai import api

"""
Pipeline Architecture:

Users/Clients (HTTP)
      │
      ▼
┌─────────────┐
│  Frontend   │  HTTP API endpoint (/generate)
└─────────────┘
      │ nova/distributed-runtime
      ▼
┌─────────────┐
│   Middle    │
└─────────────┘
      │ nova/distributed-runtime
      ▼
┌─────────────┐
│  Backend    │
└─────────────┘
"""


@service(
    resources={"cpu": "2"},
    traffic={"timeout": 30},
    nova={
        "enabled": True,
        "namespace": "inference",
    }
)
class Backend:
    def __init__(self) -> None:
        print("Starting backend")

    @nova_endpoint()
    async def generate(self, text: RequestType) -> ResponseType:
        """Generate tokens."""
        text = f"{text}-back"
        print(f"Backend received: {text}")
        for token in text.split():
            yield f"Backend: {token}"


@service(
    resources={"cpu": "2"},
    traffic={"timeout": 30},
    nova={
        "enabled": True,
        "namespace": "inference"
    }
)
class Middle:
    backend = depends(Backend)
    
    def __init__(self) -> None:
        print("Starting middle")

    @nova_endpoint()
    async def generate(self, text: str):
        """Forward requests to backend."""
        text = f"{text}-mid"
        print(f"Middle received: {text}")
        async for response in self.backend.generate(text):
            print(f"Middle received response: {response}")
            yield f"Middle: {response}"


@service(
    resources={"cpu": "1"},
    traffic={"timeout": 60}  # Regular HTTP API
)
class Frontend:
    middle = depends(Middle)
    
    def __init__(self) -> None:
        print("Starting frontend")

    @api
    async def generate(self, text: str):
        """Stream results from the pipeline."""
        print(f"Frontend received: {text}")
        async for response in self.middle.generate(text):
            yield f"Frontend: {response}" 
