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
ARG BASE_IMAGE_TAG="25.01-py3"

FROM ${BASE_IMAGE}:${BASE_IMAGE_TAG} AS triton-distributed

# TODO: non root user by default

USER root

# TODO: separate dev from runtime dependendcies

# Rust build/dev dependencies
RUN apt-get update; apt-get install -y gdb protobuf-compiler
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# 'etcd' is runtime dependency
RUN wget https://github.com/etcd-io/etcd/releases/download/v3.5.18/etcd-v3.5.18-linux-amd64.tar.gz && tar -xzf etcd-v3.5.18-linux-amd64.tar.gz
RUN cp ./etcd-v3.5.18-linux-amd64/etcd* /usr/local/bin/.

# Install OpenAI-compatible frontend and its dependencies from triton server
# repository. These are used to have a consistent interface, schema, and FastAPI
# app between Triton Core and Triton Distributed implementations.
ARG OPENAI_SERVER_TAG="r25.01"
RUN mkdir -p /opt/tritonserver/python && \
    cd /opt/tritonserver/python && \
    rm -rf openai && \
    git clone -b ${OPENAI_SERVER_TAG} --single-branch https://github.com/triton-inference-server/server.git && \
    cd server && \
    git checkout ${SERVER_OPENAI_COMMIT} && \
    cd .. && \
    mv server/python/openai openai && \
    chown -R root:root openai && \
    chmod 755 openai && \
    chmod -R go-w openai && \
    rm -rf server && \
    python3 -m pip install -r openai/requirements.txt

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
ARG GENAI_PERF_TAG="r25.01"
RUN pip install "git+https://github.com/triton-inference-server/perf_analyzer.git@${GENAI_PERF_TAG}#subdirectory=genai-perf"

# Working directory
WORKDIR /workspace

COPY runtime /workspace/runtime
RUN cd runtime/rust && \
cargo build --release --locked && cargo doc --no-deps

# Generate C bindings. Note that this is required for TRTLLM backend re-build
COPY llm /workspace/llm
RUN cd llm/rust/ && \
cargo build --release --locked && cargo doc --no-deps

# Install uv and create virtualenv for general use
COPY python-wheel /workspace/python-wheel
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
RUN mkdir -p /opt/triton && \
    uv venv /opt/triton/venv --python 3.12 && \
    source /opt/triton/venv/bin/activate && \
    cd python-wheel && \
    uv build && \
    uv pip install dist/triton_distributed_rs*cp312*.whl

# Package the bindings
RUN mkdir -p /opt/triton/llm_binding/wheels && mkdir /opt/triton/llm_binding/lib
RUN cp python-wheel/dist/triton_distributed_rs*cp312*.whl /opt/triton/llm_binding/wheels/.
RUN cp llm/rust/target/release/libtriton_llm_capi.so /opt/triton/llm_binding/lib/.
RUN cp -r llm/rust/libtriton-llm/include /opt/triton/llm_binding/.

# Backend & Framework Specific Installation
ARG FRAMEWORK="STANDARD"
ARG TENSORRTLLM_BACKEND_REPO_TAG=
ARG TENSORRTLLM_BACKEND_REBUILD=
ENV FRAMEWORK=${FRAMEWORK}
RUN --mount=type=bind,source=./container/deps/requirements.tensorrtllm.txt,target=/tmp/requirements.txt \
    --mount=type=bind,source=./container/deps/clone_tensorrtllm.sh,target=/tmp/clone_tensorrtllm.sh \
    if [[ "$FRAMEWORK" == "TENSORRTLLM" ]] ; then pip install --timeout=2000 -r /tmp/requirements.txt; /tmp/clone_tensorrtllm.sh --tensorrtllm-backend-repo-tag ${TENSORRTLLM_BACKEND_REPO_TAG} --tensorrtllm-backend-rebuild ${TENSORRTLLM_BACKEND_REBUILD} --triton-llm-path /opt/triton/llm_binding ; fi


RUN --mount=type=bind,source=./container/deps/requirements.standard.txt,target=/tmp/requirements.txt \
    if [[ "$FRAMEWORK" == "STANDARD" ]] ; then pip install --timeout=2000 -r /tmp/requirements.txt ; fi

# Backend & Framework Specific LD_LIBRARY_PATH
ARG TENSORRTLLM_FRAMEWORK
ENV FRAMEWORK_LD_LIBRARY_PATH=${TENSORRTLLM_FRAMEWORK:+/opt/tritonserver/backends/tensorrtllm/}
ENV LD_LIBRARY_PATH=${FRAMEWORK_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}
ENV TENSORRTLLM_BACKEND_REPO_TAG=$TENSORRTLLM_BACKEND_REPO_TAG
ENV TRTLLM_USE_MPI_KVCACHE=${TENSORRTLLM_FRAMEWORK:+"1"}

# TODO set VLLM Version
# ENV VLLM_VERSION
ARG VLLM_FRAMEWORK
# DEFAULT VLLM VARIABLES
# ENV VLLM_ATTENTION_BACKEND=${VLLM_FRAMEWORK:+FLASHINFER}
ENV VLLM_WORKER_MULTIPROC_METHOD=${VLLM_FRAMEWORK:+spawn}
ENV VLLM_TORCH_HOST=${VLLM_FRAMEWORK:+localhost}
ENV VLLM_TORCH_PORT=${VLLM_FRAMEWORK:+36183}
ENV VLLM_DATA_PLANE_BACKEND=${VLLM_FRAMEWORK:+nccl}
ENV VLLM_BASELINE_WORKERS=${VLLM_FRAMEWORK:+0}
ENV VLLM_CONTEXT_WORKERS=${VLLM_FRAMEWORK:+1}
ENV VLLM_GENERATE_WORKERS=${VLLM_FRAMEWORK:+1}
ENV VLLM_BASELINE_TP_SIZE=${VLLM_FRAMEWORK:+1}
ENV VLLM_CONTEXT_TP_SIZE=${VLLM_FRAMEWORK:+1}
ENV VLLM_GENERATE_TP_SIZE=${VLLM_FRAMEWORK:+1}
ENV VLLM_KV_CAPI_PATH="/opt/triton/llm_binding/lib/libtriton_llm_capi.so"
ENV PYTHONUNBUFFERED=1

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

##########################################################
# Tokenizers                                             #
##########################################################

# Todo Pull in from network and not local to repo
#
# RUN --mount=type=bind,source=./container/deps/llama3-tok.tgz,target=/tmp/llama3-tok.tgz \
#    mkdir -p /genai-perf/tokenizers && cd /genai-perf/tokenizers && tar -xzf /tmp/llama3-tok.tgz

# Working directory
WORKDIR /workspace

COPY runtime /workspace/runtime
RUN cd runtime/rust && \
    cargo build --release --locked && cargo doc --no-deps

# Install uv and create virtualenv for general use
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
RUN mkdir -p /opt/triton && \
    uv venv /opt/triton/venv --python 3.12 && \
    source /opt/triton/venv/bin/activate && \
    cd runtime/rust/python-wheel && \
    uv build && \
    uv pip install dist/triton_distributed_rs*cp312*.whl

# Install triton_distributed_rs wheel globally in container for tests that
# currently run without virtual environment activated.
# TODO: In future, we may use a virtualenv for everything and remove this.
RUN pip install /opt/triton/llm_binding/wheels/triton_distributed_rs*cp312*.whl

COPY icp /workspace/icp
RUN /workspace/icp/protos/gen_python.sh

# Install python packages
ARG PYTHON_PACKAGE_VERSION=0.0.1.dev+unknown

# SETUPTOOLS_SCM_PRETEND_VERSION allows dynamically setting the package versions during build/install.
# This allows having versioned packages during development between releases, such as commit IDs.
#
# Normally SCM version is taken directly from .git but this is not available in the Dockerfile
# and so we pass in via a buildarg
RUN SETUPTOOLS_SCM_PRETEND_VERSION_FOR_TRITON_DISTRIBUTED_ICP=${PYTHON_PACKAGE_VERSION} pip install -e /workspace/icp/python
RUN SETUPTOOLS_SCM_PRETEND_VERSION_FOR_TRITON_DISTRIBUTED_RUNTIME=${PYTHON_PACKAGE_VERSION} pip install -e /workspace/runtime/python

# Copy everything in after install steps to avoid re-running build/install
# commands on unrelated changes in other dirs.
COPY . /workspace

# Sets pythonpath for python modules
ENV PYTHONPATH="${PYTHONPATH}:/workspace/examples/python:/opt/tritonserver/python/openai/openai_frontend"

# Enable system UCX
ENV RAPIDS_LIBUCX_PREFER_SYSTEM_LIBRARY=true

# Command and Entrypoint
CMD []
ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]
