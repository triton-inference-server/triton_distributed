import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

import uvloop
from tritonserver import Metric, MetricFamily, MetricKind
from tritonserver import Server as TritonCore

from triton_distributed.icp import DataPlane, NatsEventPlane, RequestPlane
from triton_distributed.runtime import OperatorConfig as FunctionConfig
from triton_distributed.runtime import Worker
from triton_distributed.runtime.logger import get_logger
from triton_distributed.runtime.triton_core_operator import TritonCoreOperator


@dataclass
class RequestMetric:
    isl: int
    osl: int


class Metrics(TritonCoreOperator):
    def __init__(
        self,
        name: str,
        version: int,
        request_plane: RequestPlane,
        data_plane: DataPlane,
        parameters: dict,
        repository: Optional[str] = None,
        logger: logging.Logger = get_logger(__name__),
        triton_core: Optional[TritonCore] = None,
    ):
        self._triton_core = triton_core
        self._event_plane = NatsEventPlane()

        self._count_metric_family = MetricFamily(
            MetricKind.COUNTER,
            "llm_request_count",
            "Number of requests",
        )
        self._count_metric = Metric(self._count_metric_family)

        self._isl_metric_family = MetricFamily(
            MetricKind.GAUGE,
            "llm_request_isl",
            "isl",
        )
        self._isl_metric = Metric(self._isl_metric_family)

        self._osl_metric_family = MetricFamily(
            MetricKind.GAUGE,
            "llm_request_osl",
            "osl",
        )
        self._osl_metric = Metric(self._osl_metric_family)

        asyncio.create_task(self._subscribe())

    async def _callback(self, event):
        request_metric = event.typed_payload(RequestMetric)
        print(request_metric)
        self._count_metric.increment(1)
        self._isl_metric.set_value(request_metric.isl)
        self._osl_metric.set_value(request_metric.osl)

    async def _subscribe(self):
        await self._event_plane.connect()
        await self._event_plane.subscribe(
            event_topic=["request_count"], callback=self._callback
        )

    async def execute(self, requests):
        pass


def worker():
    """
    Instantiate a `backend` component and serve the `generate` endpoint
    A `Component` can serve multiple endpoints
    """
    # component = runtime.namespace("triton-init").component("vllm")
    # await component.create_service()

    # endpoint = component.endpoint("generate")
    # await endpoint.serve_endpoint(VllmEngine(engine_args).generate)

    metrics = FunctionConfig(
        name="metrics",
        implementation=Metrics,
        max_inflight_requests=0,
    )

    Worker(operators=[metrics], log_level=1, metrics_port=50000).start()


if __name__ == "__main__":
    uvloop.install()
    worker()
