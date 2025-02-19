from pydantic import BaseModel

class ServiceDecorator(BaseModel):
    name: str
    nova: "NovaConfig"
    num_replicas: int
    resources: "Resources"

class EndpointDecorator(BaseModel):
    name: str
    request: BaseModel
    response: BaseModel

class NovaConfig(BaseModel):
    enabled: bool
    namespace: str

class Resources(BaseModel):
    accelerator: str
    num_cpus: int
    num_gpus: int
