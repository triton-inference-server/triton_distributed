import typing as t
from functools import wraps
import bentoml
from typing import get_type_hints, Any
from pydantic import BaseModel

class NovaEndpoint:
    """Decorator class for Nova endpoints"""
    
    def __init__(self, func: t.Callable, name: str | None = None):
        self.func = func
        self.name = name or func.__name__
        self.is_nova_endpoint = True
        
        # Extract request type from hints
        hints = get_type_hints(func)
        args = list(hints.items())
        
        # Skip self/cls argument
        if args[0][0] in ('self', 'cls'):
            args = args[1:]
            
        # Get request type from first arg
        self.request_type = args[0][1]
        wraps(func)(self)
    
    async def __call__(self, *args: t.Any, **kwargs: t.Any) -> Any:
        # Validate request
        if len(args) > 1 and issubclass(self.request_type, BaseModel):
            args = list(args)
            if isinstance(args[1], (str, dict)):
                args[1] = self.request_type.parse_obj(args[1])
                
        # Convert Pydantic model to dict before passing to triton
        if len(args) > 1 and isinstance(args[1], BaseModel):
            args = list(args)
            args[1] = args[1].model_dump()
            
        return await self.func(*args, **kwargs)

def nova_endpoint(
    name: str | None = None
) -> t.Callable[[t.Callable], NovaEndpoint]:
    """Decorator for Nova endpoints.
    
    Args:
        name: Optional name for the endpoint. Defaults to function name.
    
    Example:
        @nova_endpoint()
        def my_endpoint(self, input: str) -> str:
            return input

        @nova_endpoint(name="custom_name")
        def another_endpoint(self, input: str) -> str:
            return input
    """
    def decorator(func: t.Callable) -> NovaEndpoint:
        return NovaEndpoint(func, name)
    
    return decorator

def api(func: t.Callable) -> t.Callable:
    """Decorator for BentoML API endpoints.
    
    Args:
        func: The function to be decorated.
    
    Returns:
        The decorated function.
    """
    return bentoml.api(func)

def async_onstart(func: t.Callable) -> t.Callable:
    """Decorator for async onstart functions.
    """
    # Mark the function as a startup hook
    setattr(func, "__bentoml_startup_hook__", True)
    return bentoml.on_startup(func)
