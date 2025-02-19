// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use super::*;

use pyo3::prelude::*;
use std::sync::Arc;

#[pyclass]
pub(crate) struct DisaggregatedRouter {
    inner: Arc<llm_rs::disagg_router::DisaggregatedRouter>,
}

#[pymethods]
impl DisaggregatedRouter {
    #[new]
    fn new(max_local_prefill_length: i32) -> PyResult<Self> {
        Ok(Self {
            inner: Arc::new(llm_rs::disagg_router::DisaggregatedRouter::new(max_local_prefill_length)),
        })
    }

    fn prefill_remote(&self, prefill_length: i32, prefix_hit_length: i32) -> bool {
        self.inner.prefill_remote(prefill_length, prefix_hit_length)
    }

    fn update_value(&mut self, max_local_prefill_length: i32) {
        Arc::get_mut(&mut self.inner)
            .expect("Cannot modify router: Arc has multiple references")
            .update_value(max_local_prefill_length);
    }

}