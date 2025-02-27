import msgspec
from vllm.sampling_params import SamplingParams


class Request(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    # required for @cached_property.
    dict=True,
):
    """The request data of one remote prefill output of a request.

    Args:
        request_id: The unique ID of the request.
        prompt: The prompt string of the request.
    """

    request_id: str
    prompt: str
    sampling_params: SamplingParams
    do_remote_prefill: bool = False
