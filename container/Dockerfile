# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG BASE_IMAGE="nvcr.io/nvidia/tritonserver"
ARG BASE_IMAGE_TAG="24.12-py3"
ARG VLLM_WHEEL

FROM ${BASE_IMAGE}:${BASE_IMAGE_TAG} AS triton-distributed

# TODO: non root user by default

USER root

# TODO: separate dev from runtime dependendcies

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

# GENAI Perf Install
ARG GENAI_PERF_TAG="r24.12"
RUN pip install "git+https://github.com/triton-inference-server/perf_analyzer.git@${GENAI_PERF_TAG}#subdirectory=genai-perf"

# Backend & Framework Specific Installation
ARG FRAMEWORK="STANDARD"
ARG TENSORRTLLM_BACKEND_COMMIT=
RUN --mount=type=bind,source=./container/deps/requirements.tensorrtllm.txt,target=/tmp/requirements.txt \
    --mount=type=bind,source=./container/deps/clone_tensorrtllm.sh,target=/tmp/clone_tensorrtllm.sh \
    if [[ "$FRAMEWORK" == "TENSORRTLLM" ]] ; then pip install --timeout=2000 -r /tmp/requirements.txt; /tmp/clone_tensorrtllm.sh --tensorrtllm-backend-commit ${TENSORRTLLM_BACKEND_COMMIT} ; fi

RUN --mount=type=bind,source=./container/deps/requirements.vllm.txt,target=/tmp/requirements.txt \
    if [[ "$FRAMEWORK" == "VLLM" ]] ; then pip install --timeout=2000 -r /tmp/requirements.txt ; fi

# NOTE: python3-distro is a system package that causes conflicts with user
# packages installed by `pip`. Removing the system package allows user packages
# to correctly manage dependencies with the `distro` pip user package instead.
RUN --mount=type=bind,source=./container/deps/,target=/tmp/deps \
    if [[ "$FRAMEWORK" == "VLLM" ]]; then \
      apt install -y zip && \
      apt remove -y python3-distro && \
      pip install distro && \
      cp -r /tmp/deps/vllm /tmp/local_vllm && \
      cd /tmp/local_vllm && \
      bash ./prepare_wheel.sh --install --debug --force ; \
    fi

RUN --mount=type=bind,source=./container/deps/requirements.standard.txt,target=/tmp/requirements.txt \
    if [[ "$FRAMEWORK" == "STANDARD" ]] ; then pip install --timeout=2000 -r /tmp/requirements.txt ; fi

# Backend & Framework Specific LD_LIBRARY_PATH
ARG TENSORRTLLM_FRAMEWORK
ENV FRAMEWORK_LD_LIBRARY_PATH=${TENSORRTLLM_FRAMEWORK:+/opt/tritonserver/backends/tensorrtllm/}
ENV LD_LIBRARY_PATH=${FRAMEWORK_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}
ENV TENSORRTLLM_BACKEND_COMMIT=$TENSORRTLLM_BACKEND_COMMIT

# TODO set VLLM Version
# ENV VLLM_VERSION

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

# Todo Pull in from network and not local to repo
#
# RUN --mount=type=bind,source=./container/deps/llama3-tok.tgz,target=/tmp/llama3-tok.tgz \
#    mkdir -p /genai-perf/tokenizers && cd /genai-perf/tokenizers && tar -xzf /tmp/llama3-tok.tgz

# Working directory
WORKDIR /workspace

#TODO Exclude container directory
COPY . /workspace

RUN /workspace/icp/protos/gen_python.sh

# Sets pythonpath for python modules
ENV PYTHONPATH="${PYTHONPATH}:/workspace/icp/src/python"

# Command and Entrypoint
CMD []
ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]
