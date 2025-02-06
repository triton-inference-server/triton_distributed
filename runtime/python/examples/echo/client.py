import asyncio

from protocol import Id, User

from triton_distributed.icp import NatsRequestPlane, UcpDataPlane

# from triton_distributed import DistributedRuntime, triton_worker
from triton_distributed.runtime import RemoteOperator


async def main():
    request_plane = NatsRequestPlane()
    await request_plane.connect()

    data_plane = UcpDataPlane()
    data_plane.connect()

    client = RemoteOperator("echo", request_plane, data_plane)

    characters = []
    async for character in client.call("hello", return_type=str):
        characters.append(character)
    print(f"Client Received Response: {''.join(characters)}")

    client = RemoteOperator("user", request_plane, data_plane)

    ids = []
    async for id_ in client.call(User(user="hello"), return_type=Id):
        ids.append(id_)
    print(f"Client Received Response: {ids[0]}")


if __name__ == "__main__":
    asyncio.run(main())
