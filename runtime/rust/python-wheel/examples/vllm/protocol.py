from pydantic import BaseModel


class Request(BaseModel):
    prompt: str
    sampling_params: dict


class PrefillRequest(Request):
    request_id: str


class Response(BaseModel):
    text: str


class PrefillResponse(BaseModel):
    prefilled: bool
