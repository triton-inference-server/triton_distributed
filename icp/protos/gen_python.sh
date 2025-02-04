#! /bin/bash
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

PROTO_SRC=$(dirname "$(realpath $0)")
SOURCE_ROOT="$(realpath "${PROTO_SRC}/..")"
PROTO_OUT=$SOURCE_ROOT/python/src/triton_distributed/icp/protos

mkdir -p $PROTO_OUT

python3 -m grpc_tools.protoc -I$PROTO_SRC --python_out=$PROTO_OUT --pyi_out=$PROTO_OUT icp.proto \
  && ls $PROTO_OUT
