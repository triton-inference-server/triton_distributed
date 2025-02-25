import argparse
import asyncio

import uvloop
from triton_distributed_rs import (
    DistributedRuntime,
    ModelDeploymentCard,
    OAIChatPreprocessor,
    triton_worker,
)

uvloop.install()


@triton_worker()
async def preprocessor(runtime: DistributedRuntime, model_name: str, model_path: str):
    # create model deployment card
    mdc = await ModelDeploymentCard.from_local_path(model_path, model_name)
    # create preprocessor endpoint
    component = runtime.namespace("triton-init").component("preprocessor")
    await component.create_service()
    endpoint = component.endpoint("generate")

    # create backend endpoint
    backend = runtime.namespace("triton-init").component("backend").endpoint("generate")

    # start preprocessor service with next backend
    chat = OAIChatPreprocessor(mdc, endpoint, next=backend)
    await chat.start()


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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(preprocessor(args.model_name, args.model_path))
