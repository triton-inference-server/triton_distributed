# wrapper over bento images to handle TritonDistributed base image

import bentoml
import os


image_name = os.getenv("NOVA_IMAGE", "nvcr.io/nvidian/nim-llm-dev/yatai-bentos:nova-base-0886e19")
NOVA_IMAGE = bentoml.images.PythonImage(base_image=image_name)
