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

# Working directory
WORKDIR /workspace

#TODO Exclude container directory
COPY . /workspace

RUN cd runtime/rust && \
    cargo build --release --locked && cargo doc --no-deps

# Generate C bindings
RUN cd runtime/rust/libtriton_llm && \
    cargo build --release --locked && cargo doc --no-deps

# Generate Python bindings
RUN cd runtime/rust/python-wheel && \
    maturin build

# Package the bindings
RUN mkdir binding_package && mkdir binding_package/wheels && mkdir binding_package/lib
RUN cp runtime/rust/python-wheel/target/wheels/*.whl binding_package/wheels/.
RUN cp runtime/rust/libtriton_llm/target/release/libtriton_llm_capi.so binding_package/lib/.
RUN cp -r runtime/rust/libtriton_llm/include binding_package/.

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

# Backend & Framework Specific Installation
ARG FRAMEWORK="STANDARD"
ARG TENSORRTLLM_BACKEND_REPO_TAG=
ARG TENSORRTLLM_BACKEND_REBUILD=
ENV FRAMEWORK=${FRAMEWORK}
RUN --mount=type=bind,source=./container/deps/requirements.tensorrtllm.txt,target=/tmp/requirements.txt \
    --mount=type=bind,source=./container/deps/clone_tensorrtllm.sh,target=/tmp/clone_tensorrtllm.sh \
    if [[ "$FRAMEWORK" == "TENSORRTLLM" ]] ; then pip install --timeout=2000 -r /tmp/requirements.txt; /tmp/clone_tensorrtllm.sh --tensorrtllm-backend-repo-tag ${TENSORRTLLM_BACKEND_REPO_TAG} --tensorrtllm-backend-rebuild ${TENSORRTLLM_BACKEND_REBUILD} --triton-llm-path /workspace/binding_package ; fi


RUN --mount=type=bind,source=./container/deps/requirements.standard.txt,target=/tmp/requirements.txt \
    if [[ "$FRAMEWORK" == "STANDARD" ]] ; then pip install --timeout=2000 -r /tmp/requirements.txt ; fi

# Backend & Framework Specific LD_LIBRARY_PATH
ARG TENSORRTLLM_FRAMEWORK
ENV FRAMEWORK_LD_LIBRARY_PATH=${TENSORRTLLM_FRAMEWORK:+/opt/tritonserver/backends/tensorrtllm/}
ENV LD_LIBRARY_PATH=${FRAMEWORK_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}
ENV TENSORRTLLM_BACKEND_REPO_TAG=$TENSORRTLLM_BACKEND_REPO_TAG
ENV TRTLLM_USE_MPI_KVCACHE=${TENSORRTLLM_FRAMEWORK:+"1"}

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
RUN mkdir /opt/triton && \
    uv venv /opt/triton/venv --python 3.12 && \
    source /opt/triton/venv/bin/activate && \
    cd runtime/rust/python-wheel && \
    uv build && \
    uv pip install dist/triton_distributed_rs*cp312*.whl

# Install patched vllm
ARG VLLM_REF="v0.7.2"
ARG VLLM_PATCH="vllm_v0.7.2.patch"
RUN --mount=type=bind,source=./container/deps/,target=/tmp/deps \
    if [[ "$FRAMEWORK" == "VLLM" ]]; then \
        source /opt/triton/venv/bin/activate && \
        bash /tmp/deps/vllm/install.sh --patch /tmp/deps/vllm/${VLLM_PATCH} --ref ${VLLM_REF} --install-cmd "uv pip install --editable" --use-precompiled --installation-dir /opt/vllm ; \
    fi


# Install triton_distributed_rs wheel globally in container for tests that
# currently run without virtual environment activated.
# TODO: In future, we may use a virtualenv for everything and remove this.
RUN pip install runtime/rust/python-wheel/dist/triton_distributed_rs*cp312*.whl

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
