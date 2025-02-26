# Wrapper for the triton_endpoint() decorator 

from models import EndpointDecorator
import compoundai
# from compoundai import nova_endpoint, api
from pydantic import BaseModel
from typing import Any, Callable

# TODO: Add support for req and resp
def nova_endpoint(name: str, **kwargs: Any) -> Callable[[type], Any]:
    return compoundai.nova_endpoint(name, **kwargs)

def nova_api() -> Callable:
    return compoundai.api()