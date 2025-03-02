import asyncio

from triton_distributed._core import DistributedRuntime


async def main():
    # Initialize runtime
    loop = asyncio.get_running_loop()
    runtime = DistributedRuntime(loop)

    # Get etcd client
    etcd = runtime.etcd_client()

    # Check what methods exist
    # print(dir(etcd))

    # Write some key-value pairs
    test_keys = {
        "test/key1": b"value1",
        "test/key2": b"value2",
        "test/nested/key3": b"value3",
    }

    # Write each key-value pair
    for key, value in test_keys.items():
        print(f"Writing {key} = {value}")
        await etcd.kv_create_or_validate(key, value, None)

    print("Successfully wrote all keys to etcd")

    # Test kv_put
    put_key = "test/put_key"
    put_value = b"put_value"
    print(f"Using kv_put to write {put_key} = {put_value}")
    await etcd.kv_put(put_key, put_value, None)

    # Test kv_get_prefix to read all keys
    print("\nReading all keys with prefix 'test/':")
    keys_values = await etcd.kv_get_prefix("test/")
    for item in keys_values:
        print(type(item["value"]))
        print(f"Retrieved {item['key']} = {item['value']}")

    # Verify prefix filtering works
    print("\nReading keys with prefix 'test/nested/':")
    nested_keys_values = await etcd.kv_get_prefix("test/nested/")
    for item in nested_keys_values:
        print(f"Retrieved {item['key']} = {item['value']}")

    # Shutdown runtime
    runtime.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
