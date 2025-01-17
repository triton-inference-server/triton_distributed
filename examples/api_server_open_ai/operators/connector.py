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
