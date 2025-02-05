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

import json
from typing import Any, Union

import numpy as np
from fastapi import Header, HTTPException


# Utility function to convert response to JSON
def tensor_to_json(tensor: np.ndarray) -> Any:
    """Convert numpy tensor to JSON."""
    if tensor.dtype.type is np.bytes_:
        items = list([item.decode("utf-8") for item in tensor.flat])
        if len(items) == 1:
            try:
                json_object = json.loads(items[0])
                return json_object
            except Exception:
                return items[0]
        return items
    return tensor.tolist()


def json_to_tensor(json_list: str) -> np.ndarray:
    """Convert JSON to numpy tensor."""
    return np.char.encode(json_list, "utf-8")


def verify_headers(content_type: Union[str, None] = Header(None)):
    """Verify content type."""
    if content_type != "application/json":
        raise HTTPException(
            status_code=415,
            detail="Unsupported media type: {content_type}. It must be application/json",
        )
