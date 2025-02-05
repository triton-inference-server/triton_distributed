import numpy

from triton_distributed.runtime import Operator, RemoteInferenceRequest, RemoteOperator


class EncodeDecodeOperator(Operator):
    def __init__(
        self,
        name,
        version,
        request_plane,
        data_plane,
        parameters,
        repository,
        logger,
        triton_core,
    ):
        self._encoder = RemoteOperator("encoder", request_plane, data_plane)
        self._decoder = RemoteOperator("decoder", request_plane, data_plane)
        self._logger = logger

    async def execute(self, requests: list[RemoteInferenceRequest]):
        self._logger.info("got request!")
        for request in requests:
            encoded_responses = await self._encoder.async_infer(
                inputs={"input": request.inputs["input"]}
            )

            async for encoded_response in encoded_responses:
                input_copies = int(
                    numpy.from_dlpack(encoded_response.outputs["input_copies"])
                )
                decoded_responses = await self._decoder.async_infer(
                    inputs={"input": encoded_response.outputs["output"]},
                    parameters={"input_copies": input_copies},
                )

                async for decoded_response in decoded_responses:
                    await request.response_sender().send(
                        final=True,
                        outputs={"output": decoded_response.outputs["output"]},
                    )
                    del decoded_response
