from pydantic import BaseModel

class Request(BaseModel):
    data: str

class Response(BaseModel):
    char: str
