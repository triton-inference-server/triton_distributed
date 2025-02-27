from compoundai.sdk.decorators import nova_endpoint, api, async_onstart, async_on_endpoint
from compoundai.sdk.service import service
from compoundai.sdk.dependency import depends
from bentoml import api
from bentoml._internal.context import server_context
from compoundai.sdk.image import NOVA_IMAGE


tdist_context = {}