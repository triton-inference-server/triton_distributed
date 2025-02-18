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

use triton_distributed::{
    error,
    icp::events::EventPlane,
    logging,
    utils::{Duration, Instant},
    DistributedRuntime, ErrorContext, Result, Runtime, Worker,
};

use tracing as log;

// enum MetricTypes {
//     LLMWorkerLoadCapacity(LLMWorkerLoadCapacityConfig),
// }

fn get_config() -> Result<LLMWorkerLoadCapacityConfig> {
    let component_name = std::env::var("TRD_COUNT_SCRAPE_COMPONENT")?;
    if component_name.is_empty() {
        return Err(error!("TRD_COUNT_SCRAPE_COMPONENT is not set"));
    }

    let endpoint_name = std::env::var("TRD_COUNT_SCRAPE_ENDPOINT")?;
    if endpoint_name.is_empty() {
        return Err(error!("TRD_COUNT_SCRAPE_ENDPOINT is not set"));
    }

    Ok(LLMWorkerLoadCapacityConfig {
        component_name,
        endpoint_name,
    })
}

// we will scrape the service_name and extract the endpoint_name metrics
// we will bcast them as {namespace}.events.l2c.{service_name}.{endpoint_name}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMWorkerLoadCapacityConfig {
    component_name: String,
    endpoint_name: String,
}

/// LLM Worker Load Capacity Metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
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

// TODO - refactor much of this back into the library
async fn app(runtime: Runtime) -> Result<()> {
    // we will start by assuming that there is no oscar and no planner
    // to that end, we will use an env to get a singular config for scraping a single backend
    let config = get_config()?;

    let drt = DistributedRuntime::from_settings(runtime.clone()).await?;

    // todo move to distributed and standardize and move into file/env/cli config
    let namespace = std::env::var("TRD_NAMESPACE").unwrap_or("default".to_string());

    let namespace = drt.namespace(namespace)?;
    let component = namespace.component("count")?;

    // there should only be one count
    // check {component.etcd_path()}/instance for existing instances
    let key = format!("{}/instance", component.etcd_path());
    drt.etcd_client()
        .kv_create(
            key,
            serde_json::to_vec_pretty(&config)?,
            Some(drt.primary_lease().id()),
        )
        .await
        .context("Unable to create unique instance of Count; possibly one already exists")?;

    let target = namespace.component(&config.component_name)?;
    let target_endpoint = target.endpoint(&config.endpoint_name);

    let service_name = target.service_name();
    let subject = target_endpoint.subject();

    log::debug!("Scraping service {service_name} and filtering on subject {subject}");
    let token = drt.primary_lease().child_token();

    let address = format!("{}.{}", config.component_name, config.endpoint_name,);
    let subject = format!("l2c.{}", address);

    loop {
        // TODO - make this configurable
        let next = Instant::now() + Duration::from_secs(2);
        let stream = target.scrape_stats(Duration::from_secs(1)).await?;

        let endpoints = stream
            .into_endpoints()
            .filter(|e| e.subject.starts_with(&subject))
            .collect::<Vec<_>>();

        let metrics = endpoints
            .iter()
            .filter_map(|e| match e.data.clone() {
                Some(metrics) => metrics.decode::<LLMWorkerLoadCapacity>().ok(),
                None => None,
            })
            .collect::<Vec<_>>();

        // parse the endpoint ids
        // the ids are the last part of the subject in hexadecimal
        // form a list of tuples (kv_blocks_total - kv_blocks_active, requests_total_slots - requests_active_slots, id)
        let capacity_with_ids = metrics
            .iter()
            .zip(endpoints.iter())
            .filter_map(|(m, e)| {
                e.id().ok().map(|id| {
                    (
                        m.kv_blocks_total - m.kv_blocks_active,
                        m.requests_total_slots - m.requests_active_slots,
                        id,
                    )
                })
            })
            .collect::<Vec<_>>();

        // compute mean / std of LLMWorkerLoadCapacity
        let load_values: Vec<f64> = metrics.iter().map(|x| x.kv_blocks_active as f64).collect();
        let load_avg = load_values.iter().sum::<f64>() / load_values.len() as f64;
        let variance = load_values
            .iter()
            .map(|&x| (x - load_avg).powi(2))
            .sum::<f64>()
            / load_values.len() as f64;
        let load_std = variance.sqrt();

        let processed = ProcessedEndpoints {
            capacity_with_ids,
            load_avg,
            load_std,
            address: address.clone(),
        };

        // publish using the namespace event plane
        namespace.publish(&subject, &processed).await?;

        // wait until cancelled or the next tick
        match tokio::time::timeout_at(next, token.cancelled()).await {
            Ok(_) => break,
            Err(_) => {
                // timeout, we continue
                continue;
            }
        }
    }

    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessedEndpoints {
    /// (kv_blocks_total - kv_blocks_active, requests_total_slots - requests_active_slots, id)
    pub capacity_with_ids: Vec<(u32, u32, i64)>,

    /// kv_blocks_active average
    pub load_avg: f64,

    /// kv_blocks_active standard deviation
    pub load_std: f64,

    /// {component}.{endpoint}
    pub address: String,
}
