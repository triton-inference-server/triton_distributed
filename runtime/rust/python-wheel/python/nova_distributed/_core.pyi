from typing import AsyncGenerator, AsyncIterator, Callable

class JsonLike:
    """
    Any PyObject which can be serialized to JSON
    """
    ...

RequestHandler = Callable[[JsonLike], AsyncGenerator[JsonLike, None]]


class DistributedRuntime:
    """
    The runtime object for a distributed NOVA applications
    """
    ...

    def namespace(self, name: str, path: str) -> Namespace:
        """
        Create a `Namespace` object
        """
        ...

class Namespace:
    """
    A namespace is a collection of components
    """
    ...

    def component(self, name: str) -> Component:
        """
        Create a `Component` object
        """
        ...

class Component:
    """
    A component is a collection of endpoints
    """
    ...

    def create_service(self) -> None:
        """
        Create a service
        """
        ...

    def endpoint(self, name: str) -> Endpoint:
        """
        Create an endpoint
        """
        ...

class Endpoint:
    """
    An Endpoint is a single API endpoint
    """
    ...

    async def serve_endpoint(self, handler: RequestHandler) -> None:
        """
        Serve an endpoint discoverable by all connected clients at 
        `{{ namespace }}/components/{{ component_name }}/endpoints/{{ endpoint_name }}`
        """
        ...

    async def client() -> Client:
        """
        Create a `Client` capable of calling served instances of this endpoint
        """
        ...

class Client:
    """
    A client capable of calling served instances of an endpoint
    """
    ...

    async def random(self, request: JsonLike) -> AsyncIterator[JsonLike]:
        """
        Pick a random instance of the endpoint and issue the request
        """
        ...

    async def round_robin(self, request: JsonLike) -> AsyncIterator[JsonLike]:
        """
        Pick the next instance of the endpoint in a round-robin fashion
        """
        ...

    async def direct(self, request: JsonLike, instance: str) -> AsyncIterator[JsonLike]:
        """
        Pick a specific instance of the endpoint
        """
        ...
