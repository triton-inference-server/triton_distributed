import os
from contextlib import contextmanager

import etcd3
import msgspec
from vllm.distributed.device_communicators.nixl import NixlMetadata

METADATA_DIR = "/tmp/nixl"


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


class NixlMetadataStore:
    NIXL_METADATA_KEY = "nixl_metadata"

    def __init__(self, namespace: str) -> None:
        self._namespace = namespace
        self._stored = set()
        self._cached = {}
        self._client = etcd3.client()
        self._key_prefix = f"{self._namespace}/{NixlMetadataStore.NIXL_METADATA_KEY}"

    def _watch_callback(self, event):
        pass

    def put(self, engine_id, metadata: NixlMetadata):
        serialized_metadata = msgspec.msgpack.encode(metadata)
        key = "/".join([self._key_prefix, engine_id])
        self._client.put(key, serialized_metadata)

        self._stored.add(engine_id)

    def get(self, engine_id) -> NixlMetadata:
        if engine_id in self._cached:
            return self._cached[engine_id]

        value, metadata = self._client.get(
            f"{self._namespace}/{NixlMetadataStore.NIXL_METADATA_KEY}/{engine_id}"
        )

        # print("got value", value,metadata)

        print("got value")

        try:
            deserialized_metadata = msgspec.msgpack.decode(value, type=NixlMetadata)
        except Exception as e:
            print(e)

        print("got deserialized value")

        #        print(deserialized_metadata)

        self._cached[engine_id] = deserialized_metadata

        self._client.add_watch_callback(
            f"{self._namespace}/{NixlMetadataStore.NIXL_METADATA_KEY}/{engine_id}",
            self._watch_callback,
        )

        return deserialized_metadata
