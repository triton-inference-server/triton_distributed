# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

ARG BASE_IMAGE="nvcr.io/nvidia/tritonserver"
ARG BASE_IMAGE_TAG="24.12-py3"

FROM ${BASE_IMAGE}:${BASE_IMAGE_TAG} AS triton-distributed

# TODO: non root user by default

USER root

RUN apt-get update; apt-get install -y gdb

# Common dependencies
RUN --mount=type=bind,source=./container/deps/requirements.txt,target=/tmp/requirements.txt \
    pip install --timeout=2000 --requirement /tmp/requirements.txt
RUN --mount=type=bind,source=./container/deps/requirements.nats.txt,target=/tmp/requirements.txt \
    pip install --timeout=2000 --requirement /tmp/requirements.txt
RUN --mount=type=bind,source=./container/deps/requirements.test.txt,target=/tmp/requirements.txt \
    pip install --timeout=2000 --requirement /tmp/requirements.txt

# Finish pyright install
RUN pyright --help > /dev/null 2>&1

# In Process Python API Install
RUN find /opt/tritonserver/python -maxdepth 1 -type f -name \
    "tritonserver-*.whl" | xargs -I {} pip3 install --force-reinstall --upgrade {}[all]

# Triton CLI Install
#ARG TRITON_CLI_TAG="0.0.10"
#RUN pip install git+https://github.com/triton-inference-server/triton_cli.git@${TRITON_CLI_TAG}

# Installing triton-cli downgrades protobuf and tritonclient
# TODO find a better way to handle this
# consider moving to poetry
#RUN pip install --upgrade protobuf==5.27.3 tritonclient==2.49.0

# GENAI Perf Install
ARG GENAI_PERF_TAG="nnshah1-triton-generate"
RUN pip install "git+https://github.com/triton-inference-server/perf_analyzer.git@${GENAI_PERF_TAG}#subdirectory=genai-perf"

# Backend & Framework Specific Installation
ARG FRAMEWORK="IDENTITY"
ARG TRTLLM_BACKEND_COMMIT=
ARG GITLAB_TOKEN=
RUN --mount=type=bind,source=./container/deps/requirements.tensorrtllm.txt,target=/tmp/requirements.txt \
    --mount=type=bind,source=./container/deps/clone_tensorrtllm.sh,target=/tmp/clone_tensorrtllm.sh \
    if [[ "$FRAMEWORK" == "TENSORRTLLM" ]] ; then pip install --timeout=2000 -r /tmp/requirements.txt; /tmp/clone_tensorrtllm.sh --trtllm-backend-commit ${TRTLLM_BACKEND_COMMIT} --gitlab-token ${GITLAB_TOKEN}; fi

RUN --mount=type=bind,source=./container/deps/requirements.vllm.txt,target=/tmp/requirements.txt \
    if [[ "$FRAMEWORK" == "VLLM" ]] ; then pip install --timeout=2000 -r /tmp/requirements.txt ; fi

RUN --mount=type=bind,source=./container/deps/requirements.identity.txt,target=/tmp/requirements.txt \
    if [[ "$FRAMEWORK" == "IDENTITY" ]] ; then pip install --timeout=2000 -r /tmp/requirements.txt ; fi

# Backend & Framework Specific LD_LIBRARY_PATH
ARG TENSORRTLLM_FRAMEWORK
ENV FRAMEWORK_LD_LIBRARY_PATH=${TENSORRTLLM_FRAMEWORK:+/opt/tritonserver/backends/tensorrtllm/}
ENV LD_LIBRARY_PATH=${FRAMEWORK_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}

# Install router dependencies
RUN if [[ "$FRAMEWORK" != "IDENTITY" ]] ; then \
    apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential cmake \
    git clone https://oauth2:${GITLAB_TOKEN}@gitlab-master.nvidia.com/jothomson/kv-aware-router.git \
    cd kv-aware-router && git fetch &&  git checkout rm-tekit-submodule  && git submodule update --init --recursive && pip install -e . ;\
    fi

# Install NATS - pointing toward NATS github instead of binaries.nats.dev due to server instability
RUN wget https://github.com/nats-io/nats-server/releases/download/v2.10.24/nats-server-v2.10.24-amd64.deb && dpkg -i nats-server-v2.10.24-amd64.deb

# Enable Git operations in the /workspace directory.
RUN printf "[safe]\n      directory=/workspace\n" > /root/.gitconfig

# emacs docker-tramp requires /bin/sh to be linked to bash to operate correctly
RUN ln -sf /bin/bash /bin/sh

# Install NGINX
RUN apt-get install nginx -y
RUN rm -rf /etc/nginx/sites-enabled/default

# Install demo utils
RUN apt-get install nvtop -y
RUN apt-get install tmux -y

# Install OpenAI-compatible frontend and its dependencies
# NOTE: Has a couple compat fixes needed for kserve frontends in 24.11 release.
#       Can be replaced with specific release like r24.12 in future for stability.
ARG SERVER_OPENAI_COMMIT="f336fa6bd5416ba2f17e5eb7de5228213562bbc8"
WORKDIR /opt/tritonserver
RUN git clone https://github.com/triton-inference-server/server.git && \
    cd server && \
    git checkout ${SERVER_OPENAI_COMMIT} && \
    cd .. && \
    mv server/python/openai openai && \
    chown -R root:root openai && \
    chmod 755 openai && \
    chmod -R go-w openai && \
    rm -rf server && \
    python3 -m pip install -r openai/requirements.txt

##########################################################
# Tokenizers                                             #
##########################################################

RUN --mount=type=bind,source=./container/deps/llama3-tok.tgz,target=/tmp/llama3-tok.tgz \
    mkdir -p /genai-perf/tokenizers && cd /genai-perf/tokenizers && tar -xzf /tmp/llama3-tok.tgz

# Working directory
WORKDIR /workspace

# Sets pythonpath for python modules
ENV PYTHONPATH="${PYTHONPATH}:/workspace/icp/python:/workspace/icp/python/icp/protos:/workspace/worker/python:/workspace/triton/api_server/src/:/workspaces/tekit_backend/tensorrt_llm"

# Command and Entrypoint
CMD []
ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]
