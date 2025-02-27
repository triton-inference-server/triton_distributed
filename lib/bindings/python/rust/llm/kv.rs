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
use async_once_cell::OnceCell as AsyncOnceCell;

use triton_distributed_runtime::{DistributedRuntime, Worker};
use triton_distributed_llm::kv_router::{
    indexer::compute_block_hash_for_seq, protocols::*, publisher::KvEventPublisher,
};

static WK: OnceCell<Worker> = OnceCell::new();
static DRT: AsyncOnceCell<DistributedRuntime> = AsyncOnceCell::new();
static KV_PUB: OnceCell<KvEventPublisher> = OnceCell::new();

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

    fn create_service<'p>(
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


#[pyfunction]
pub fn triton_llm_event_init(
    namespace: String,
    component: String,
    worker_id: i64,
) -> PyResult<TritonLlmResult> {
    let wk = rs::Worker::from_settings().map_err(to_pyerr)?;
    INIT.get_or_try_init(|| {
        let primary = worker.tokio_runtime()?;
        pyo3_async_runtimes::tokio::init_with_runtime(primary)
            .map_err(|e| rs::error!("failed to initialize runtime: {:?}", e))?;
        rs::OK(())
    })
    .map_err(to_pyerr)?;

    let rt = wk.runtime().clone();
    let secondary = rt.secondary().clone();
    let result = secondary.block_on(async {
        // Initialize the distributed runtime
        match DRT
            .get_or_try_init(async { DistributedRuntime::from_settings(rt.clone()).await })
            .await
        {
            Ok(_) => Ok(()),
            Err(e) => {
                map_err!("Failed to initialize distributed runtime: {:?}", e);
                Err(TritonLlmResult::ERR)
            }
        }
    });
    let namespace = match namespace {
        Ok(s) => s.to_string(),
        Err(e) => {
            eprintln!("Failed to convert C string to Rust string: {:?}", e);
            return TritonLlmResult::ERR;
        }
    };

    let component = match component {
        Ok(s) => s.to_string(),
        Err(e) => {
            eprintln!("Failed to convert C string to Rust string: {:?}", e);
            return TritonLlmResult::ERR;
        }
    };

    match result {
        Ok(_) => match KV_PUB
            .get_or_try_init(move || triton_create_kv_publisher(namespace, component, worker_id))
        {
            Ok(_) => TritonLlmResult::OK,
            Err(e) => {
                eprintln!("Failed to initialize distributed runtime: {:?}", e);
                TritonLlmResult::ERR
            }
        },
        Err(e) => e,
    }
}


fn triton_create_kv_publisher(
    namespace: String,
    component: String,
    worker_id: i64,
) -> Result<KvEventPublisher, anyhow::Error> {
    tracing::info!("Creating KV Publisher for model: {}", component);
    match DRT
        .get()
        .ok_or(anyhow::Error::msg("Could not get Distributed Runtime"))
    {
        Ok(drt) => {
            let backend = drt.namespace(namespace)?.component(component)?;
            KvEventPublisher::new(drt.clone(), backend, worker_id)
        }
        Err(e) => Err(e),
    }
}