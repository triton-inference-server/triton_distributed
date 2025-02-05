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

from typing import Type


class CustomKeyErrorDict(dict):
    def __init__(
        self,
        from_name: str,
        to_name: str,
        *args,
        exception: Type[Exception] = ValueError,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._to_name = to_name
        self._from_name = from_name
        self._exception = exception

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            raise self._exception(
                f"Unsupported {self._from_name}. Can't convert {key} to {self._to_name}"
            ) from None
