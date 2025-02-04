import asyncio
import uvloop

from nova_distributed import nova_worker, DistributedRuntime, HttpAsyncEngine, HttpService 

## WARNING: This is a WIP and does not currently work as expected


@nova_worker()
async def worker(runtime: DistributedRuntime):
    model_client = await runtime.namespace("my-model").component("llm-backend").endpoint("generate").client()
    await model_client.generate("yo")
    loop = runtime.event_loop()

    async def hi():
        print("hi")
        await asyncio.sleep(1)
        print("hi")

    await loop.create_task(hi())

    # there is some weird bug here where the loop is dropped or not copied correctly
    # the http service gets an error when calling the model engine's .generate call
    # `RuntimeError: no running event loop`
    #
    # the all-in-rust version of this works fine, so it's some python-rust interop issue
    #
    # see apps/nova/http-ingress for the rust code that works

    # this is the fix, but does not solve the issue or provide the correct semantics for 
    # the distrubuted's AsyncEngine contract
    #
    # the contract for the distributed pipeline system is that all .generate calls are async
    # and they must await and downstream callers call chain
    #
    # this pattern break that chain and causes the error handling in the request-call-chain
    # to be deferred to the stream handler, which is not the behavior in the contract
    #
    # we will want to shore up the contract via the PythonAsyncEngine object and enforce
    # that the pyobject passed to the PythonAsyncEngine wrapper is a future to an async generator
    #
    # We can raise an exception in rust if the call to the python object is not a future
    def client_wrapper(request):
        async def generator():
            stream = await model_client.generate(request)
            async for char in stream:
                yield char
        return generator()

    http_engine = HttpAsyncEngine(client_wrapper, loop)
    await model_client.generate("yo")

    http = HttpService(9992)
    http.add_chat_completions_model("my-model", http_engine)
    #http_service.mark_as_ready()

    task = http.run(runtime.child_token())

    await asyncio.sleep(120)

    runtime.shutdown()

    await task

    #await http.run(runtime.child_token())

#asyncio.run(worker())

if __name__ == "__main__":
    loop = asyncio.new_event_loop()

    port = 8887
    http_service = HttpService(port=port)

    loop.run_until_complete(worker())
