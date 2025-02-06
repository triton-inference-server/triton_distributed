import time
from typing import AsyncIterator

from protocol import Id, User

from triton_distributed.icp import NatsServer
from triton_distributed.runtime import CallableOperator, OperatorConfig, Worker


class User:
    """
    Request handler for the generate operator
    """

    async def user(self, user: User) -> AsyncIterator[Id]:
        print(f"Server Received request: {user}")
        yield Id("foo")


class Echo:
    """
    Request handler for the generate operator
    """

    async def echo(self, request: str) -> AsyncIterator[str]:
        print(f"Server Received request: {request}")
        for char in request:
            yield char


def worker():
    echo_op = OperatorConfig(
        name="echo",
        implementation=CallableOperator,
        parameters={"callable_object": Echo().echo},
    )

    user_op = OperatorConfig(
        name="user",
        implementation=CallableOperator,
        parameters={"callable_object": User().user},
    )

    Worker(operators=[echo_op, user_op], log_level=1).start()


if __name__ == "__main__":
    request_plane_server = NatsServer(log_dir=None)
    time.sleep(2)
    worker()
