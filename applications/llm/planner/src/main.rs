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

//! Overwatch is a top-level service that monitors the state of a single Nova Init
//! Deployment.
//!
//! Primary responsibilities:
//! - Monitor each component of the pipeline is marked ready
//! - For components which are expected to expose an [Endpoint], ensure that at least one
//!   instance is ready to receive traffic
//!
//! - Pipeline state:
//!   - Setup
//!   - Ready
//!   - Unavailable
//!   - TearDown
//!
//! - Actions:
//!   - Setup:
//!     - Start all components
//!     - Observe state from backend/terminus to frontend
//!       - Order provided by the init-graph
//!   - Healthy:
//!     - Customization point
//!       - the primary action when health for an llm pipeline is to register the pipeline
//!         as a model name with the http ingress
//!       - specialiations of this service can perform other actions on transition to healthy
//!   - Unhealthy:
//!     - Customization point
//!       - the primary action when an llm pipeline is unhealthy is to mark the model in the http
//!         ingress as unhealthy so it can return a 503 Service Unavailable error
//!       - specialiations of this service can perform other actions on transition to unhealthy
//!   - TearDown:
//!     - Stop all components
//!     - For a Nova Init deployment, the following actions are taken:
//!       - The model is permanently removed from the http ingress ensuring no new requests are
//!         forwarded to the pipeline.
//!       - Each pipeline component is asked to gracefully terminate. This will allow each outstanding
//!         task/stream to try to complete.
//!       - Components like routers can be terminated after all frontend are marked to the TearDown
//!         state, which ensures that no new requests will be accepted by any of the frontend.
//!       - Any persistent state in ETCD, NATS, MinIO, etc. should be removed.
//!       - If this process were to die/fail, persistent state might not be removed, which is a problem
//!         that the global Oscar service will detect and correct.
//!
//! Overwatch should be able to write to a special path in ETCD to track each instance regardless of
//! which namespace it belongs. This reserved path will allow for tools to quickly parse the set of active
//! pipelines and read their top-level configs and state.
//!

// TODO - remove after implementation
#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]
//
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing as log;
use trd::{error, logging, DistributedRuntime, ErrorContext, Result, Runtime, Worker};
use triton_distributed::{self as trd, actions::Action, engine::async_trait};
use triton_llm::http::service::actions::HttpAction;

fn main() -> Result<()> {
    logging::init();
    let worker = Worker::from_settings()?;
    worker.execute(app)
}

// TODO - refactor much of this back into the library
async fn app(runtime: Runtime) -> Result<()> {
    let drt = DistributedRuntime::from_settings(runtime.clone()).await?;
    let id = drt.primary_lease().id();

    log::debug!("Overwatch ID: {}", id);

    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasicInitGraph {
    namespace: String,

    /// A map of component name to a list of endpoints
    /// A [Component] have 0 or more endpoints
    /// Each [Component] will have a [ServiceState] which will be monitored.
    /// If a [Component] has [Endpoints][Endpoint], then the list of workers for each [Endpoint]
    /// will also be monitored.
    components: HashMap<String, Vec<String>>,
}

/// Action triggered on Setup
/// This action will process the InitGraph and perform the coordindated bringup of all component
/// and endpoints in reverse dependency order, i.e. from the backend to the frontend.
pub struct InitGraphSetupAction {}

/// Action triggered on Cleanup
/// This action will process the InitGraph and perform the coordindated tear down of all components
/// and endpoints.
///
/// This action will immediately remove the model from the http ingress.
pub struct InitGraphCleanupAction {}
