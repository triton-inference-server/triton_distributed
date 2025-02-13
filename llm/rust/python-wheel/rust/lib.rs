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

use triton_distributed_py3;
use pyo3::{exceptions::PyException, prelude::*};
use std::{fmt::Display, sync::Arc};

use triton_llm as rs;

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<KvRouter>()?;

    Ok(())
}

pub fn to_pyerr<E>(err: E) -> PyErr
where
    E: Display,
{
    PyException::new_err(format!("{}", err))
}

#[pyclass]
pub(crate) struct KvRouter {
    inner: Arc<rs::kv_router::KvRouter>,
}

#[pymethods]
impl KvRouter {
    #[new]
    fn new(drt: triton_distributed_py3::DistributedRuntime, component: triton_distributed_py3::Component) -> PyResult<Self> {
        let runtime = pyo3_async_runtimes::tokio::get_runtime();
        runtime.block_on(async {
            let inner =
                rs::kv_router::KvRouter::from_runtime(drt.inner.clone(), component.inner.clone())
                    .await
                    .map_err(to_pyerr)?;
            Ok(Self { inner })
        })
    }

    fn schedule<'p>(
        &self,
        py: Python<'p>,
        token_ids: Vec<u32>,
        lora_id: u64,
    ) -> PyResult<Bound<'p, PyAny>> {
        let router = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let uuid = router
                .schedule(&token_ids, lora_id)
                .await
                .map_err(to_pyerr)?;
            Ok(uuid.to_string())
        })
    }
}
