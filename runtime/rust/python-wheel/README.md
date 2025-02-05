<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Python Bindings for Rust

This README is under construction and will have more details added over time.

```bash
cd triton_distributed/runtime/rust/python-wheel

# Build python package for bindings (requires cargo and maturin)
maturin build

# Install the python package for binded rust code
pip install target/wheels/triton_distributed_rs-*.whl

# Sanity check the package installed successfully
python3 -c "import triton_distributed_rs"
```
