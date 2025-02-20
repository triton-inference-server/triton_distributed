import typing as t
from functools import wraps
import bentoml

class NovaEndpoint:
    """Decorator class for Nova endpoints"""
    
    def __init__(self, func: t.Callable, name: str | None = None):
        self.func = func
        self.name = name or func.__name__
        # Mark this as a Nova endpoint for discovery
        self.is_nova_endpoint = True
        # Copy function metadata
        wraps(func)(self)
    
    def __call__(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        return self.func(*args, **kwargs)

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
