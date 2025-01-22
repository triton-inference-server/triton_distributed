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
from .connector import MyTritonDistributedConnector
from triton_distributed.worker import Operator, RemoteInferenceRequest


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


class ApiServerOperator(Operator):
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

        self.openai_frontend.start()
        while True:
            time.sleep(1)

        # The simplest approach: spawn uvicorn in a background thread
        self.server_thread = None
        self.should_stop = False
        self.server_thread = threading.Thread(target=self.start_server)
        self.server_thread.join()

    async def execute(self, requests: list[RemoteInferenceRequest]):
        """
        This can remain effectively no-op for typical requests. Or it can do
        something if you want to handle requests from the request plane.

        But mostly, the purpose is that once the worker is started, we
        spawn the server. The requests come in via HTTP, not the request plane.
        """
        self._logger.info(
            "API Server operator ignoring direct requests, it's purely for hosting HTTP endpoints."
        )
        for req in requests:
            await req.response_sender().send(final=True)  # or respond with NotSupported

    def start_server(self):
        """
        Launch uvicorn in a background thread or so
        """
        self._logger.info(
            "API Server thread starts"
        )
        raise Exception("EVIL")
        self.openai_frontend.start()
        self._logger.info(
            "API Server thread start finished and sleeping will start"
        )
        while True:
            time.sleep(1)
        self._logger.info(
            "API Server thread sleep finished"
        )
