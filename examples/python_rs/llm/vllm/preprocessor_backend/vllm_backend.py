import argparse
import asyncio
import inspect
import uuid

import torch
import uvloop
from triton_distributed_rs import (
    Backend,
    DistributedRuntime,
    ModelDeploymentCard,
    triton_worker,
)
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from vllm.outputs import CompletionOutput
from vllm.utils import FlexibleArgumentParser

uvloop.install()


finish_reason_map = {
    None: None,
    "stop": "stop",
    "abort": "cancelled",
    "length": "length",
    "error": "error",
}


class DeltaState:
    """
    The vLLM AsyncEngine returns the full internal state of each slot per forward pass.
    The OpenAI ChatCompletionResponseDelta object only requires the delta, so this object
    is used to track the state of the last forward pass to calculate the delta.
    """

    def __init__(self):
        self.token_ids = None
        self.last_token_count = 0

    def delta(self, choice):
        self.token_ids = choice.token_ids
        tokens_produced = len(choice.token_ids) - self.last_token_count
        self.last_token_count = len(choice.token_ids)
        return choice.token_ids[-tokens_produced:]


class RequestHandler:
    """
    Request handler for the generate endpoint
    """

    def __init__(self, async_engine_args: AsyncEngineArgs, mdc: ModelDeploymentCard):
        self.mdc = mdc
        self.engine = AsyncLLMEngine.from_engine_args(async_engine_args)
        print("vllm backend started")

    def to_backend_output(self, response: CompletionOutput, delta_token_ids: list[int]):
        return {
            "token_ids": delta_token_ids,
            "tokens": [],
            "finish_reason": finish_reason_map.get(response.finish_reason, "stop"),
            "cum_log_probs": response.cumulative_logprob,
            "text": None,
        }

    def to_sampling_params(self, request) -> SamplingParams:
        sampling_params_names = inspect.signature(SamplingParams).parameters.keys()
        sampling_params = {
            k: v
            for k, v in request.get("sampling_options", {}).items()
            if k in sampling_params_names and v is not None
        }
        return SamplingParams(**sampling_params)

    async def generate(self, request):
        print(f"received request: {request}")
        state = DeltaState()
        request_id = str(uuid.uuid4())
        sampling_params = self.to_sampling_params(request)
        inputs = {"prompt_token_ids": request["token_ids"]}
        print(f"sampling_params: {sampling_params}")
        stream = self.engine.generate(inputs, sampling_params, request_id=request_id)

        async for request_output in stream:
            for choice in request_output.outputs:
                delta_token_ids = state.delta(choice)
                yield self.to_backend_output(choice, delta_token_ids)


@triton_worker()
async def worker(runtime: DistributedRuntime, args: argparse.Namespace):
    """
    Instantiate a `backend` component and serve the `generate` endpoint
    A `Component` can serve multiple endpoints
    """
    mdc = await ModelDeploymentCard.from_local_path(args.model_path, args.model)
    component = runtime.namespace("triton-init").component("backend")
    await component.create_service()
    print(f"args: {args}")
    async_engine_args = AsyncEngineArgs(
        model=args.model,
        dtype="auto",
        skip_tokenizer_init=True,  # ensure the tokenizer is not initialized.
        enforce_eager=True,
        disable_log_stats=True,
        disable_async_output_proc=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    handler = RequestHandler(async_engine_args, mdc)
    endpoint = component.endpoint("generate")
    backend = Backend(mdc, endpoint)
    await backend.start(handler.generate)


def parse_vllm_args():
    parser = FlexibleArgumentParser()
    parser = AsyncEngineArgs.add_cli_args(parser)
    parser.add_argument(
        "--model-path",
        type=str,
        default="~/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    uvloop.install()
    args = parse_vllm_args()
    asyncio.run(worker(args))
