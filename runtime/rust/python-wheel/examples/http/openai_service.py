import uuid
import time

import asyncio
import uvloop

from nova_distributed import nova_worker, DistributedRuntime, HttpService, HttpAsyncEngine


class DummyEngine:
    def __init__(self, model_name):
        self.model_name = model_name

    def generate(self, request):
        id = f"chat-{uuid.uuid4()}"
        created = int(time.time())
        model = self.model_name
        print(f"{created} | Received request: {request}")

        async def generator():
            num_chunks = 5
            for i in range(num_chunks):
                dummy_content = f"chunk{i}"
                finish_reason = "stop" if (i == num_chunks-1) else None

                chunk = {
                    "id": id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{
                        "index": i,
                        "delta": {"role": None, "content": dummy_content},
                        "logprobs": None,
                        "finish_reason": finish_reason
                    }]
                }
                yield chunk

        return generator()


@nova_worker()
async def worker(runtime: DistributedRuntime):
    model: str = "dummy_model"
    served_model_name: str = "dummy_model"

    loop = asyncio.get_running_loop()
    python_engine = DummyEngine(model)
    engine = HttpAsyncEngine(python_engine.generate, loop)

    host: str = "localhost"
    port: int = 8000
    service: HttpService = HttpService(port=port)
    service.add_chat_completions_model(served_model_name, engine)

    print("Starting service...")
    shutdown_signal = service.run(runtime.child_token())

    try:
        print(f"Serving endpoint: {host}:{port}/v1/models")
        print(f"Serving endpoint: {host}:{port}/v1/chat/completions")
        print(f"Serving the following models: {service.list_chat_completions_models()}")
        # Block until shutdown signal received
        await shutdown_signal
    except KeyboardInterrupt:
        # FIXME: Caught by DistributedRuntime or HttpService, so not caught here
        pass
    except Exception as e:
        print(f"Unexpected error occurred: {e}")
    finally:
        print("Shutting down worker...")
        runtime.shutdown()

if __name__ == "__main__":
    uvloop.install()
    # FIXME: linter complains about lack of runtime arg passed
    asyncio.run(worker())
