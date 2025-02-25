import argparse
import asyncio
import logging

import uvloop
from triton_distributed_rs import (
    DistributedRuntime,
    HttpAsyncEngine,
    HttpService,
    triton_worker,
)

logging.basicConfig(level=logging.INFO)


class OpenAIChatService:
    def __init__(self, model_name, model_path, preprocessor):
        self.model_name = model_name
        self.model_path = model_path
        self.preprocessor = preprocessor

    async def generate(self, request):
        print(f"Received request: {request}")
        logging.info(f"Received request: {request}")
        async for resp in await self.preprocessor.random(request):
            logging.info(f"Sending response: {resp}")
            yield resp["data"]


@triton_worker()
async def worker(
    runtime: DistributedRuntime, model_name: str, model_path: str, port: int
):
    loop = asyncio.get_running_loop()
    preprocessor = (
        await runtime.namespace("triton-init")
        .component("preprocessor")
        .endpoint("generate")
        .client()
    )
    python_engine = OpenAIChatService(model_name, model_path, preprocessor)

    engine = HttpAsyncEngine(python_engine.generate, loop)

    host: str = "localhost"
    service: HttpService = HttpService(port=port)
    service.add_chat_completions_model(model_name, engine)

    logging.info("Starting service...")
    shutdown_signal = service.run(runtime.child_token())
    try:
        logging.info(f"Serving endpoint: {host}:{port}/v1/models")
        # TODO: add completion endpoint
        logging.info(
            f"Serving chat completion endpoint: {host}:{port}/v1/chat/completions"
        )
        logging.info(
            f"Serving the following models: {service.list_chat_completions_models()}"
        )
        # Block until shutdown signal received
        await shutdown_signal
    except KeyboardInterrupt:
        # FIXME: Caught by DistributedRuntime or HttpService, so not caught here
        pass
    except Exception as e:
        logging.error(f"Unexpected error occurred: {e}")
    finally:
        logging.info("Shutting down worker...")
        runtime.shutdown()


## Add arg parse to parse the model name and port


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="~/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    )
    # only used by http service
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


if __name__ == "__main__":
    uvloop.install()
    args = parse_args()
    asyncio.run(worker(args.model_name, args.model_path, args.port))
