# wrapper over bento images to handle TritonDistributed base image

import bentoml

NOVA_IMAGE = bentoml.images.PythonImage(base_image="triton-distributed:cai")
