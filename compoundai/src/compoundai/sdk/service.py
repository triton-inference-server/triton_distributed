from typing import TypeVar, Any, Optional, Dict, Tuple, List
from dataclasses import dataclass
from _bentoml_sdk import Service, ServiceConfig
from _bentoml_sdk.images import Image
from compoundai.sdk.decorators import NovaEndpoint

from typing_extensions import TypedDict


class NovaSchema(TypedDict, total=False):
    """Configuration for Nova components"""
    enabled: bool
    name: Optional[str]
    namespace: Optional[str]


class CompoundServiceConfig(ServiceConfig):
    """Extended ServiceConfig that includes Nova-specific configuration"""
    nova: NovaSchema

T = TypeVar("T", bound=object)

class CompoundService(Service[T]):
    """A custom service class that extends BentoML's base Service with Nova capabilities"""

    def __init__(
        self,
        config: CompoundServiceConfig,
        inner: type[T],
        image: Optional[Image] = None,
        envs: List[Dict[str, Any]] = None,
    ):
        super().__init__(config=config, inner=inner, image=image, envs=envs or [])
        
        # Set defaults on nova object
        # TODO: Implement with dataclass or some other way with better validation
        self._nova_config = config.get("nova", {})
        self._nova_config["name"] = self._nova_config.get("name", inner.__name__)
        self._nova_config["namespace"] = self._nova_config.get("namespace", "default")
        self._nova_config["enabled"] = self._nova_config.get("enabled", False)

        # Register Nova endpoints.
        self._nova_endpoints: Dict[str, NovaEndpoint] = {}
        for field in dir(inner):
            value = getattr(inner, field)
            if isinstance(value, NovaEndpoint):
                self._nova_endpoints[value.name] = value

    def is_nova_component(self) -> bool:
        """Check if this service is configured as a Nova component"""
        return self._nova_config["enabled"]

    def nova_address(self) -> Tuple[str, str]:
        """Get the Nova address for this component in namespace/name format"""
        if not self.is_nova_component():
            raise ValueError("Service is not configured as a Nova component")
        return (self._nova_config["namespace"], self._nova_config["name"])

    def get_nova_endpoints(self) -> Dict[str, NovaEndpoint]:
        """Get all registered Nova endpoints"""
        return self._nova_endpoints

    def get_nova_endpoint(self, name: str) -> NovaEndpoint:
        """Get a specific Nova endpoint by name"""
        if name not in self._nova_endpoints:
            raise ValueError(f"No Nova endpoint found with name: {name}")
        return self._nova_endpoints[name]

    def list_nova_endpoints(self) -> List[str]:
        """List names of all registered Nova endpoints"""
        return list(self._nova_endpoints.keys())



def service(
    inner: Optional[type[T]] = None,
    /,
    *,
    image: Optional[Image] = None,
    envs: Optional[List[Dict[str, Any]]] = None,
    **kwargs: Any,
) -> Any:
    """Enhanced service decorator that supports Nova configuration

    Args:
        nova: Nova configuration, either as a NovaSchema dict or an object with keys:
            - enabled: bool (default False)
            - name: str (default: class name)
            - namespace: str (default: "default")
        **kwargs: Existing BentoML service configuration
    """
    config = kwargs
    print("kwargs", kwargs)

    def decorator(inner: type[T]) -> CompoundService[T]:
        if isinstance(inner, Service):
            raise TypeError("service() decorator can only be applied once")
        return CompoundService(
            config=config,
            inner=inner,
            image=image,
            envs=envs or [],
        )

    return decorator(inner) if inner is not None else decorator
