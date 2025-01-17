# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
            except:
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
