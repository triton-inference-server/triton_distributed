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

#[pyclass]
pub(crate) struct KvRouter {
    inner: Arc<llm_rs::kv_router::KvRouter>,
}

#[pymethods]
impl KvRouter {
    #[new]
    // [FXIME] 'drt' can be obtained from 'component'
    fn new(drt: DistributedRuntime, component: Component) -> PyResult<Self> {
        let runtime = pyo3_async_runtimes::tokio::get_runtime();
        runtime.block_on(async {
            let inner = llm_rs::kv_router::KvRouter::from_runtime(
                drt.inner.clone(),
                component.inner.clone(),
            )
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
            let worker_id = router
                .schedule(&token_ids, lora_id)
                .await
                .map_err(to_pyerr)?;
            Ok(worker_id)
        })
    }
}

#[pyclass]
pub(crate) struct KvMetricsPublisher {
    inner: Arc<llm_rs::kv_router::publisher::KvMetricsPublisher>,
}

#[pymethods]
impl KvMetricsPublisher {
    #[new]
    fn new() -> PyResult<Self> {
        let inner = llm_rs::kv_router::publisher::KvMetricsPublisher::new().map_err(to_pyerr)?;
        Ok(Self {
            inner: inner.into(),
        })
    }

    fn create_endpoint<'p>(
        &self,
        py: Python<'p>,
        component: Component,
    ) -> PyResult<Bound<'p, PyAny>> {
        let rs_publisher = self.inner.clone();
        let rs_component = component.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            rs_publisher
                .create_endpoint(rs_component)
                .await
                .map_err(to_pyerr)?;
            Ok(())
        })
    }

    fn publish(
        &self,
        _py: Python,
        request_active_slots: u64,
        request_total_slots: u64,
        kv_active_blocks: u64,
        kv_total_blocks: u64,
    ) -> PyResult<()> {
        self.inner
            .publish(
                llm_rs::kv_router::protocols::ForwardPassMetrics {
                    request_active_slots,
                    request_total_slots,
                    kv_active_blocks,
                    kv_total_blocks,
                }
                .into(),
            )
            .map_err(to_pyerr)
    }
}
