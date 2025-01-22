import argparse
import asyncio
import shutil
import signal
import sys
from pathlib import Path

from triton_distributed.worker import Deployment, OperatorConfig, WorkerConfig



import asyncio
import threading
import time
from typing import AsyncIterator, List

# FIXME: Make openai_frontend components a proper installable package
# Implementing LLMEngine interface from openai_frontend in triton repo that
# are pulled into container/environment and added to PYTHONPATH at build time:
# https://github.com/triton-inference-server/server/blob/2ebd762fa6c7b829e7d04bfaf80c8400a09d3767/python/openai/openai_frontend/engine/engine.py#L41
from engine.engine import LLMEngine
from frontend.fastapi_frontend import FastApiFrontend
from schemas.openai import (
    CreateChatCompletionRequest,
    CreateChatCompletionResponse,
    CreateCompletionRequest,
    CreateCompletionResponse,
    Model,
    ObjectType,
)
from triton_api_server.open_ai.chat_vllm import ChatHandlerVllm
#from .connector import MyTritonDistributedConnector
from triton_distributed.worker import Operator, RemoteInferenceRequest


import typing

from triton_api_server.connector import (
    BaseTriton3Connector,
    InferenceRequest,
    InferenceResponse,
)
from triton_distributed.worker import RemoteOperator


class MyTritonDistributedConnector(BaseTriton3Connector):
    def __init__(self, request_plane, data_plane):
        # store them
        self._request_plane = request_plane
        self._data_plane = data_plane

    async def inference(
        self, model_name: str, request: InferenceRequest
    ) -> typing.AsyncGenerator[InferenceResponse, None]:
        # create a remote operator with name==model_name?
        # or do some switch logic:
        remote_op = RemoteOperator(model_name, self._request_plane, self._data_plane)
        # build a RemoteInferenceRequest from request
        remote_request = remote_op.create_request(
            inputs=request.inputs, parameters=request.parameters
        )
        # call async_infer
        async for resp in await remote_op.async_infer(inference_request=remote_request):
            # transform to InferenceResponse
            yield InferenceResponse(outputs=..., final=resp.final, error=resp.error)
            # you'll want to parse resp.outputs and fill them in


class TritonDistributedChatHandler(ChatHandlerVllm):
    def __init__(
        self,
        triton_connector: MyTritonDistributedConnector,
        model_name: str,
        tokenizer: str,
    ):
        super().__init__(triton_connector, model_name, tokenizer)

    # Request / response format can vary between frontends, so allow override
    # of adaptor functions accordingly.
    def stream_response_adaptor(self, response_stream):
        async def adaptor_stream():
            async for response in response_stream():
                if isinstance(response, Exception):
                    raise response
                else:
                    # Already in SSE String format
                    yield response

        return adaptor_stream

    def response_adaptor(self, response):
        return response

    def exception_adaptor(self, exception):
        raise exception


class TritonDistributedEngine(LLMEngine):
    def __init__(
        self,
        request_plane,
        data_plane,
        # nats_url: str,
        # data_plane_host: str,
        # data_plane_port: int,
        model_name: str,
        tokenizer: str,
    ):
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.triton_connector = MyTritonDistributedConnector(request_plane, data_plane)
        # self.triton_connector = RemoteModelConnector(
        #     nats_url=nats_url,
        #     data_plane_host=data_plane_host,
        #     data_plane_port=data_plane_port,
        #     model_name=model_name,
        #     keep_dataplane_endpoints_open=True,
        # )

        # FIXME: Consider supporting multiple or per-model tokenizers
        self.request_handler = TritonDistributedChatHandler(
            self.triton_connector, model_name, tokenizer
        )

    async def chat(
        self, request: CreateChatCompletionRequest
    ) -> CreateChatCompletionResponse | AsyncIterator[str]:
        """
        If request.stream is True, this returns an AsyncIterator (or Generator) that
        produces server-sent-event (SSE) strings in the following form:
            'data: {CreateChatCompletionStreamResponse}\n\n'
            ...
            'data: [DONE]\n\n'

        If request.stream is False, this returns a CreateChatCompletionResponse.
        """
        # FIXME: Unify call whether streaming or not
        if request.stream:
            response_generator = await self.request_handler.process_request(
                request, None
            )
            return response_generator()

        response = await self.request_handler.process_request(request, None)
        return response

    async def completion(
        self, request: CreateCompletionRequest
    ) -> CreateCompletionResponse | AsyncIterator[str]:
        """
        If request.stream is True, this returns an AsyncIterator (or Generator) that
        produces server-sent-event (SSE) strings in the following form:
            'data: {CreateCompletionResponse}\n\n'
            ...
            'data: [DONE]\n\n'

        If request.stream is False, this returns a CreateCompletionResponse.
        """
        raise NotImplementedError

    def ready(self) -> bool:
        """
        Returns True if the engine is ready to accept inference requests, or False otherwise.
        """
        # FIXME: Add more useful checks if available.
        return True

    def metrics(self) -> str:
        """
        Returns the engine's metrics in a Prometheus-compatible string format.
        """
        raise NotImplementedError

    def models(self) -> List[Model]:
        """
        Returns a List of OpenAI Model objects.
        """
        # FIXME: Support 'async def models()'
        model_names = asyncio.run(self.triton_connector.list_models())

        models = [
            Model(
                id=model_name,
                object=ObjectType.model,
                owned_by="Triton Distributed",
                # FIXME: Need to track creation times, so set to 0 for now.
                created=0,
            )
            for model_name in model_names
        ]

        return models


# def main():
#     args = parse_args()
#     logging.basicConfig(level=args.log_level.upper(), format=args.log_format)

#     # Wrap Triton Distributed in an interface-conforming "LLMEngine"
#     engine: TritonDistributedEngine = TritonDistributedEngine(
#         nats_url=args.nats_url,
#         data_plane_host=args.data_plane_host,
#         data_plane_port=args.data_plane_port,
#         model_name=args.model_name,
#         tokenizer=args.tokenizer,
#     )

#     # Attach TritonLLMEngine as the backbone for inference and model management
#     openai_frontend: FastApiFrontend = FastApiFrontend(
#         engine=engine,
#         host=args.api_server_host,
#         port=args.api_server_port,
#         log_level=args.log_level.lower(),
#     )

#     # Blocking call until killed or interrupted with SIGINT
#     openai_frontend.start()


# if __name__ == "__main__":
#     main()


class ApiServerOperator:
    def __init__(
        self,
        name,
        version,
        triton_core,
        request_plane,
        data_plane,
        parameters,
        repository,
        logger,
    ):
        self._request_plane = request_plane
        self._data_plane = data_plane
        self._params = parameters
        self._logger = logger

        # Wrap Triton Distributed in an interface-conforming "LLMEngine"
        self.engine: TritonDistributedEngine = TritonDistributedEngine(
            request_plane=self._request_plane,
            data_plane=self._data_plane,
            # nats_url=parameters.nats_url,
            # data_plane_host=parameters.data_plane_host,
            # data_plane_port=parameters.data_plane_port,
            model_name=parameters["model_name"],
            tokenizer=parameters["tokenizer"],
        )

        # Attach TritonLLMEngine as the backbone for inference and model management
        self.openai_frontend: FastApiFrontend = FastApiFrontend(
            engine=self.engine,
            host=parameters["api_server_host"],
            port=parameters["api_server_port"],
            log_level=parameters["log_level"].lower(),
        )

    def run(self):
        self.openai_frontend.start()


def parse_args(args=None):
    example_dir = Path(__file__).parent.absolute().parent.absolute()

    default_log_dir = example_dir.joinpath("logs")

    parser = argparse.ArgumentParser(description="Hello World Deployment")

    parser.add_argument(
        "--initialize-request-plane",
        default=False,
        action="store_true",
        help="Initialize the request plane, should only be done once per deployment",
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        default=str(default_log_dir),
        help="log dir folder",
    )

    parser.add_argument(
        "--clear-logs", default=False, action="store_true", help="clear log dir"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        required=False,
        default="DEBUG",
        help="Logging level",
    )

    # parser.add_argument(
    #     "--request-plane-uri", type=str, default="nats://localhost:4223"
    # )

    # API Server
    parser.add_argument(
        "--api-server-host",
        type=str,
        required=False,
        default="127.0.0.1",
        help="API Server host",
    )

    parser.add_argument(
        "--api-server-port",
        type=int,
        required=False,
        default=8000,
        help="API Server port",
    )

    # Misc
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Tokenizer to use for chat template in chat completions API",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        required=False,
        default="prefill",
        help="Model name",
    )

    parser.add_argument(
        "--log-format",
        type=str,
        required=False,
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        help="Logging format",
    )

    args = parser.parse_args(args)

    return args


deployment = None


def handler(signum, frame):
    exit_code = 0
    if deployment:
        print("Stopping Workers")
        exit_code = deployment.stop()
    print(f"Workers Stopped Exit Code {exit_code}")
    sys.exit(exit_code)


signals = (signal.SIGHUP, signal.SIGTERM, signal.SIGINT)
for sig in signals:
    try:
        signal.signal(sig, handler)
    except Exception:
        pass


def main(args):

    global deployment
    log_dir = Path(args.log_dir)

    if args.clear_logs:
        shutil.rmtree(log_dir)

    log_dir.mkdir(exist_ok=True)

    parameters={
        "api_server_host": args.api_server_host,
        "api_server_port": args.api_server_port,
        "tokenizer": args.tokenizer,
        "model_name": args.model_name,
        "log_level": args.log_level,
        "log_format": args.log_format,
    }


    # define all your worker configs as before: encoder, decoder, etc.
    api_server_op = OperatorConfig(
        name="api_server",
        implementation="api_server_open_ai.operators.api_server_operator:ApiServerOperator",
        parameters={
            "api_server_host": args.api_server_host,
            "api_server_port": args.api_server_port,
            "tokenizer": args.tokenizer,
            "model_name": args.model_name,
            "log_level": args.log_level,
            "log_format": args.log_format,
        },
        max_inflight_requests=1,
    )

    api_server = ApiServerOperator(name=None,
        version=None,
        triton_core=None,
        request_plane=None,
        data_plane=None,
        parameters=parameters,
        repository=None,
        logger=None,
    )
    api_server.run()



    # api_server = WorkerConfig(operators=[api_server_op], name="api_server")

    # deployment = Deployment(
    #     [
    #         (api_server, 1),
    #     ],
    #     initialize_request_plane=True,
    #     log_dir=args.log_dir,
    #     log_level=args.log_level,
    # )
    # deployment.start()
    # while True:
    #     await asyncio.sleep(10)


if __name__ == "__main__":
    args = parse_args()
    main(args)
