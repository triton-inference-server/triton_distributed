import asyncio
import enum
import logging
import os
from contextlib import nullcontext

import torch

from .connector import InferenceRequest
from .remote_model_connector import RemoteModelConnector
from .request_converter import RequestConverter

LOGGER = logging.getLogger(__name__)


class _ProfileState(enum.Enum):
    NOT_STARTED = 0
    STARTED = 1
    STOPPED = 2


class PiplineStageExecutor:
    def __init__(self, args, stage, stage_name, next_stage_name=None):
        self.args = args
        self.stage = stage
        self.stage_name = stage_name
        self.is_context_stage = next_stage_name is not None
        self.next_stage_name = next_stage_name
        self.remote_model_connector = (
            RemoteModelConnector(
                nats_url=self.args.nats_url,
                model_name=self.next_stage_name,
                keep_dataplane_endpoints_open=True,
            )
            if self.is_context_stage
            else None
        )
        self.request_converter = RequestConverter(
            nats_url=self.args.nats_url,
            keep_dataplane_endpoints_open=True,
            model_name=self.args.worker_name,
        )
        self.request_counter = 0
        self.profile_state = _ProfileState.NOT_STARTED
        self.tasks = []

    async def baseline_process(self, request, return_result):
        try:
            LOGGER.debug("Processing request")
            async for response in self.stage(request):
                LOGGER.debug("Sending response")
                await return_result(**response)
                LOGGER.debug("Response send")
        except Exception as e:
            LOGGER.error(f"Error processing request: {e}")
            await return_result({"error": e, "final": True})
        LOGGER.debug("Processing finished")

    async def process(self, request, return_result):
        LOGGER.debug("Processing request")
        try:
            LOGGER.debug(f"Stage {self.stage_name} execution")
            responses = list([response async for response in self.stage(request)])
            LOGGER.debug(f"Stage {self.stage_name} finished")
            assert len(responses) == 1
            response = responses[0]

            parameters = response.get("parameters", {})
            if not parameters:
                raise RuntimeError(
                    f"ERROR: Response parameters from stage {self.stage_name} should not be empty!"
                )

            outputs = response.get("outputs", {})
            request = InferenceRequest(inputs=outputs, parameters=parameters)
            LOGGER.info(f"Next stage {self.next_stage_name} execution")
            assert self.remote_model_connector is not None
            async for response in self.remote_model_connector.inference(
                model_name=self.next_stage_name, request=request
            ):
                LOGGER.debug(f"Stage {self.stage_name} sending response")
                await return_result(
                    outputs=response.outputs,
                    final=response.final,
                    parameters={"text": response.parameters["text"]},
                )
                LOGGER.debug(f"Stage {self.stage_name} sended response")
        except Exception as e:
            LOGGER.error(f"Error processing request: {e}", exc_info=True)
            await return_result(outputs={}, error=e, final=True)

    async def handle_pipelined_requests(self):
        LOGGER.info(
            f"Start handling requests stage_name {self.stage_name} args {self.args}"
        )
        async with self.request_converter, self.remote_model_connector or nullcontext():
            LOGGER.info(f"Stage {self.stage_name} starts pulling")
            async for request, return_result in self.request_converter.pull(
                model_name=self.args.worker_name
            ):
                # TODO ptarasiewicz - only one context or generate should be profiled at a time
                await self.process_request(request, return_result)
            LOGGER.info(f"Stage {self.stage_name} finished pulling")

    async def process_requests(self, requests):
        for raw_request in requests:
            (
                inputs,
                remote_request,
                return_callable,
            ) = await self.request_converter.adapt_request(raw_request)

            request, return_result = {
                "inputs": inputs,
                "parameters": remote_request.parameters,
            }, return_callable
            await self.process_request(request, return_result)

    async def process_request(self, request, return_result):
        self._profile()
        if self.is_context_stage:
            process_function = self.process
        else:
            process_function = self.baseline_process
        # self.request_counter += 1
        LOGGER.debug(f"Stage {self.stage_name} pulled request")
        self.tasks.append(asyncio.create_task(process_function(request, return_result)))
        if len(self.tasks) >= self.args.max_batch_size:
            LOGGER.debug(
                f"Stage {self.stage_name} waiting some of {len(self.tasks)} requests to finish"
            )
            _, pending = await asyncio.wait(
                self.tasks, return_when=asyncio.FIRST_COMPLETED
            )
            self.tasks = list(pending)
            LOGGER.debug(
                f"Stage {self.stage_name} finished some requests with {len(self.tasks)} to do"
            )

    def _profile(self):
        if os.environ.get("RUN_PROFILING") == "1":
            if (
                self.profile_state == _ProfileState.NOT_STARTED
                and self.request_counter > 100
            ):
                LOGGER.info("Start profiling")
                torch.cuda.profiler.start()
                self.profile_state = _ProfileState.STARTED
            elif (
                self.profile_state == _ProfileState.STARTED
                and self.request_counter > 120
            ):
                LOGGER.info("Stop profiling")
                torch.cuda.profiler.stop()
                self.profile_state = _ProfileState.STOPPED
            # can also use with torch.cuda.profiler.profile():
