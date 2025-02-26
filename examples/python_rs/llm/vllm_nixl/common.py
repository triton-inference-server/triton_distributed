import os
from contextlib import contextmanager

import msgspec
from vllm.distributed.device_communicators.nixl import NixlMetadata
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.utils import FlexibleArgumentParser

METADATA_DIR = "/tmp/nixl"


def parse_vllm_args() -> AsyncEngineArgs:
    parser = FlexibleArgumentParser()
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    return AsyncEngineArgs.from_cli_args(args)


@contextmanager
def temp_metadata_file(engine_id, metadata: NixlMetadata):
    os.makedirs(METADATA_DIR, exist_ok=True)
    path = f"{METADATA_DIR}/{engine_id}.nixl_meta"
    with open(path, "wb") as f:
        encoded = msgspec.msgpack.encode(metadata)
        print(f"Size of encoded metadata: {len(encoded)}")
        f.write(encoded)
    try:
        yield path
    finally:
        if os.path.exists(path):
            os.remove(path)


def find_remote_metadata(engine_id):
    # find and load metadata from METADATA_DIR that do not match engine_id
    remote_metadata = []
    for file in os.listdir(METADATA_DIR):
        if file.endswith(".nixl_meta"):
            if file.split(".")[0] != engine_id:
                with open(os.path.join(METADATA_DIR, file), "rb") as f:
                    remote_metadata.append(
                        msgspec.msgpack.decode(f.read(), type=NixlMetadata)
                    )
    return remote_metadata
