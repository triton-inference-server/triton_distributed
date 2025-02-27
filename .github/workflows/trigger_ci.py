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

import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", type=str, required=True)
    parser.add_argument("--filters", type=str, required=True)
    parser.add_argument("--commit-message", type=str, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(args)
