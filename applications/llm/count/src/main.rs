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

//! Count is a metrics aggregator designed to operate within a namespace and collect
//! metrics from all workers.
//!
//! Metrics will collect for now:
//!
//! - LLM Worker Load:Capacity
//!   - These metrics will be scraped by the LLM NATS Service API's stats request
//!   - Request Slots: [Active, Total]
//!   - KV Cache Blocks: [Active, Total]

use serde::{Deserialize, Serialize};

use triton_distributed::{logging, DistributedRuntime, Result, Runtime, Worker};

enum MetricTypes {
    LLMWorkerLoadCapacity(LLMWorkerLoadCapacityConfig),
}

// we will scrape the service_name and extract the endpoint_name metrics
// we will bcast them as {namespace}.events.l2c.{service_name}.{endpoint_name}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMWorkerLoadCapacityConfig {
    service_name: String,
    endpoint_name: Vec<String>,
}

/// LLM Worker Load Capacity Metrics
pub struct LLMWorkerLoadCapacity {
    pub requests_active_slots: u32,
    pub requests_total_slots: u32,
    pub kv_blocks_active: u32,
    pub kv_blocks_total: u32,
}

fn main() -> Result<()> {
    logging::init();
    let worker = Worker::from_settings()?;
    worker.execute(app)
}

async fn app(runtime: Runtime) -> Result<()> {
    let distributed = DistributedRuntime::from_settings(runtime.clone()).await?;

    // todo move to distributed and standardize and move into file/env/cli config
    let namespace = std::env::var("NOVA_NAMESPACE").unwrap_or("default".to_string());

    let namespace = distributed.namespace(namespace)?;
    let component = namespace.component("count")?;

    // there should only be one count
    // check {component.etcd_path()}/instance for existing instances
    // if not, present, atomically create one
    // if present, error

    // read and watch {component.etcd_path()}/config for the set of data we will aggregate
    // for now, just wait until the config is present or use the current value
    // longer term, adapt to updates to the config file

    // for each metric type, kick of the collection task

    Ok(())
}
