# Wrapper for the triton_service() decorator

from models import ServiceDecorator
import compoundai
from typing import Any, Callable

def nova_service(namespace: str, **kwargs: Any) -> Callable[[type], Any]:
    """Custom decorator for Nova services with default config"""
    # Merge user-provided kwargs with Nova defaults
    nova_config = {
        "enabled": True,
        "namespace": namespace,  # default namespace
        **kwargs
    }
    
    # Pass through to BentoML's service decorator
    return compoundai.service(nova=nova_config)
