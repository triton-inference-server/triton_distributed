from pydantic import BaseModel

class Request(BaseModel):
    prompt: str
    sampling_params: dict


class Response(BaseModel):
    text: str