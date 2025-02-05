# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest

try:
    import vllm
except ImportError:
    vllm = None  # type: ignore

pytestmark = pytest.mark.pre_merge


# TODO: Consider `pytest.mark.vllm` and running tests based on environment
@pytest.mark.skipif(vllm is None, reason="Skipping vllm tests, vllm not installed")
def test_version():
    # Verify that the image has the patched version of vllm
    assert vllm.__version__ == "0.6.3.post2.dev16+gf61960ce"


@pytest.mark.skipif(vllm is None, reason="Skipping vllm tests, vllm not installed")
def test_patch_imports():
    # Verify patched files have no glaring syntax or import issues
    import vllm.distributed.data_plane as d
    import vllm.distributed.kv_cache as k

    # Placeholder to avoid unused import errors or removal by linters
    assert d, k
