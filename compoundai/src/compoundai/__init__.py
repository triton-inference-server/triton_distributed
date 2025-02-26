from bentoml import api
from bentoml._internal.context import server_context

from compoundai.sdk.decorators import api, async_onstart, nova_endpoint
from compoundai.sdk.dependency import depends
from compoundai.sdk.image import NOVA_IMAGE
from compoundai.sdk.service import service

tdist_context = {}
