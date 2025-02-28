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

use clap::Parser;
use prometheus::register_gauge_vec;
use serde::{Deserialize, Serialize};
use warp::Filter;

// Import the types from the KV router library
use triton_distributed_llm::kv_router::protocols::ForwardPassMetrics;
use triton_distributed_llm::kv_router::scheduler::Endpoint;
use triton_distributed_llm::kv_router::scoring::ProcessedEndpoints;

use triton_distributed_runtime::{
    distributed::Component,
    error, logging,
    service::EndpointInfo,
    traits::events::EventPublisher,
    utils::{Duration, Instant},
    DistributedRuntime, ErrorContext, Result, Runtime, Worker,
};

/// CLI arguments for the count application
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Component to scrape metrics from
    #[arg(long)]
    component: String,

    /// Endpoint to scrape metrics from
    #[arg(long)]
    endpoint: String,

    /// Namespace to operate in
    #[arg(long, env = "TRD_NAMESPACE", default_value = "triton-init")]
    namespace: String,

    /// Polling interval in seconds (minimum 1 second)
    #[arg(long, default_value = "2")]
    poll_interval: u64,
}

fn get_config(args: &Args) -> Result<LLMWorkerLoadCapacityConfig> {
    if args.component.is_empty() {
        return Err(error!("Component name cannot be empty"));
    }

    if args.endpoint.is_empty() {
        return Err(error!("Endpoint name cannot be empty"));
    }

    if args.poll_interval < 1 {
        return Err(error!("Polling interval must be at least 1 second"));
    }

    Ok(LLMWorkerLoadCapacityConfig {
        component_name: args.component.clone(),
        endpoint_name: args.endpoint.clone(),
    })
}

// we will scrape the service_name and extract the endpoint_name metrics
// we will bcast them as {namespace}.events.l2c.{service_name}.{endpoint_name}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMWorkerLoadCapacityConfig {
    component_name: String,
    endpoint_name: String,
}

// FIXME: The object returned from scraping stats is _almost_ the
// async_nats::service::endpoint::Stats object, but is missing some fields
// like "name". Define a custom struct for deserializing into for now.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct StatsWithData {
    average_processing_time: f64,
    data: serde_json::Value,
    last_error: String,
    num_errors: u64,
    num_requests: u64,
    processing_time: u64,
    queue_group: String,
}

// Define a struct to hold all Prometheus metrics and server
struct PrometheusMetricsServer {
    // Encapsulate the metrics in a struct
    metrics: PrometheusMetrics,
}

impl PrometheusMetricsServer {
    // Initialize all metrics
    fn new() -> Result<Self> {
        Ok(Self {
            metrics: PrometheusMetrics::new()?,
        })
    }

    // Start the metrics server
    fn start(&mut self, port: u16) {
        let metrics_route = warp::path!("metrics").map(|| {
            use prometheus::Encoder;
            let encoder = prometheus::TextEncoder::new();
            let mut buffer = Vec::new();
            encoder.encode(&prometheus::gather(), &mut buffer).unwrap();
            String::from_utf8(buffer).unwrap()
        });

        let server = warp::serve(metrics_route).run(([0, 0, 0, 0], port));
        tokio::spawn(server);
        tracing::info!("Prometheus metrics server started on port {}", port);
    }

    // Update metrics with current values
    fn update(&mut self, config: &LLMWorkerLoadCapacityConfig, processed: &ProcessedEndpoints) {
        self.metrics.update(config, processed);
    }
}

// TODO: Should prometheus metrics move into library with ForwardPassMetrics?
struct PrometheusMetrics {
    kv_blocks_active: prometheus::GaugeVec,
    kv_blocks_total: prometheus::GaugeVec,
    requests_active: prometheus::GaugeVec,
    requests_total: prometheus::GaugeVec,
    load_avg: prometheus::GaugeVec,
    load_std: prometheus::GaugeVec,
}

impl PrometheusMetrics {
    // Initialize all metrics
    fn new() -> Result<Self> {
        Ok(Self {
            kv_blocks_active: register_gauge_vec!(
                "llm_kv_blocks_active",
                "Active KV cache blocks",
                &["component", "endpoint", "worker_id"]
            )?,
            kv_blocks_total: register_gauge_vec!(
                "llm_kv_blocks_total",
                "Total KV cache blocks",
                &["component", "endpoint", "worker_id"]
            )?,
            requests_active: register_gauge_vec!(
                "llm_requests_active_slots",
                "Active request slots",
                &["component", "endpoint", "worker_id"]
            )?,
            requests_total: register_gauge_vec!(
                "llm_requests_total_slots",
                "Total request slots",
                &["component", "endpoint", "worker_id"]
            )?,
            load_avg: register_gauge_vec!(
                "llm_load_avg",
                "Average load across workers",
                &["component", "endpoint"]
            )?,
            load_std: register_gauge_vec!(
                "llm_load_std",
                "Load standard deviation across workers",
                &["component", "endpoint"]
            )?,
        })
    }

    // Update metrics with current values
    fn update(&self, config: &LLMWorkerLoadCapacityConfig, processed: &ProcessedEndpoints) {
        // Update per-worker metrics
        for endpoint in processed.endpoints.iter() {
            let worker_id = endpoint.worker_id().to_string();
            let metrics = endpoint.data.clone();
            self.kv_blocks_active
                .with_label_values(&[&config.component_name, &config.endpoint_name, &worker_id])
                .set(metrics.kv_active_blocks as f64);

            self.kv_blocks_total
                .with_label_values(&[&config.component_name, &config.endpoint_name, &worker_id])
                .set(metrics.kv_total_blocks as f64);

            self.requests_active
                .with_label_values(&[&config.component_name, &config.endpoint_name, &worker_id])
                .set(metrics.request_active_slots as f64);

            self.requests_total
                .with_label_values(&[&config.component_name, &config.endpoint_name, &worker_id])
                .set(metrics.request_total_slots as f64);
        }

        // Update aggregate metrics
        self.load_avg
            .with_label_values(&[&config.component_name, &config.endpoint_name])
            .set(processed.load_avg);

        self.load_std
            .with_label_values(&[&config.component_name, &config.endpoint_name])
            .set(processed.load_std);
    }
}

// Helper function to collect endpoints from a component
async fn collect_endpoints(
    target: &Component,
    service_subject: &str,
    timeout: Duration,
) -> Result<Vec<EndpointInfo>> {
    // Collect stats from each backend
    let stream = target.scrape_stats(timeout).await?;
    tracing::debug!("Scraped Stats Stream: {stream:?}");

    // Filter the stats by the service subject
    let endpoints = stream
        .into_endpoints()
        .filter(|e| e.subject.starts_with(service_subject))
        .collect::<Vec<_>>();

    tracing::debug!("Endpoints: {endpoints:?}");
    if endpoints.is_empty() {
        tracing::warn!("No endpoints found matching subject {}", service_subject);
    }

    Ok(endpoints)
}

// Helper function to extract metrics from endpoints
fn extract_metrics(
    endpoints: &[triton_distributed_runtime::service::EndpointInfo],
) -> Vec<ForwardPassMetrics> {
    let endpoint_data = endpoints.iter().map(|e| e.data.clone()).collect::<Vec<_>>();
    tracing::debug!("Endpoint Data: {endpoint_data:?}");

    // Extract StatsWithData
    let stats: Vec<StatsWithData> = endpoint_data
        .iter()
        .filter_map(|e| {
            let metrics_data = e.as_ref()?;
            serde_json::from_value::<StatsWithData>(metrics_data.0.clone()).ok()
        })
        .collect();
    tracing::debug!("Stats: {stats:?}");

    // TODO: Make this more general to various types of metrics
    // Extract ForwardPassMetrics nested within Stats object
    let metrics: Vec<ForwardPassMetrics> = stats
        .iter()
        .filter_map(
            |s| match serde_json::from_value::<ForwardPassMetrics>(s.data.clone()) {
                Ok(metrics) => Some(metrics),
                Err(err) => {
                    tracing::warn!("Error decoding metrics: {err}");
                    None
                }
            },
        )
        .collect();
    tracing::debug!("Metrics: {metrics:?}");

    metrics
}

// Helper function to create ProcessedEndpoints
fn postprocess_metrics(
    metrics: &[ForwardPassMetrics],
    endpoints: &[EndpointInfo],
) -> ProcessedEndpoints {
    let endpoints_for_router: Vec<Endpoint> = metrics
        .iter()
        .zip(endpoints.iter())
        .filter_map(|(m, e)| {
            e.id().ok().map(|id| Endpoint {
                name: format!("worker-{}", id),
                subject: e.subject.clone(),
                data: m.clone(),
            })
        })
        .collect();

    ProcessedEndpoints::new(endpoints_for_router)
}

async fn app(runtime: Runtime) -> Result<()> {
    let args = Args::parse();
    // we will start by assuming that there is no oscar and no planner
    // to that end, we will use CLI args to get a singular config for scraping a single backend
    let config = get_config(&args)?;
    tracing::info!("Config: {config:?}");

    let drt = DistributedRuntime::from_settings(runtime.clone()).await?;

    let namespace = drt.namespace(args.namespace)?;
    let component = namespace.component("count")?;

    // there should only be one count
    // check {component.etcd_path()}/instance for existing instances
    let key = format!("{}/instance", component.etcd_path());
    tracing::info!("Creating unique instance of Count at {key}");
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
    let service_subject = target_endpoint.subject();
    tracing::info!("Scraping service {service_name} and filtering on subject {service_subject}");

    let token = drt.primary_lease().child_token();

    let address = format!("{}.{}", config.component_name, config.endpoint_name,);
    let event_name = format!("l2c.{}", address);

    // Initialize Prometheus metrics and start server
    let mut metrics_server = PrometheusMetricsServer::new()?;
    metrics_server.start(9091);

    loop {
        let next = Instant::now() + Duration::from_secs(args.poll_interval);
        let endpoints =
            collect_endpoints(&target, &service_subject, Duration::from_secs(1)).await?;
        let metrics = extract_metrics(&endpoints);
        let postprocessed = postprocess_metrics(&metrics, &endpoints);

        // Update Prometheus metrics
        metrics_server.update(&config, &postprocessed);

        // TODO: Who should consume these events?
        // Publish metrics event
        tracing::debug!(
            "Publishing event {event_name} on namespace {namespace:?} with {postprocessed:?}"
        );
        namespace.publish(&event_name, &postprocessed).await?;

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

fn main() -> Result<()> {
    logging::init();
    let worker = Worker::from_settings()?;
    worker.execute(app)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_namespace_from_env() {
        env::set_var("TRD_NAMESPACE", "test-namespace");

        // Parse args with no explicit namespace
        let args = Args::parse_from(["count", "--component", "comp", "--endpoint", "end"]);

        // Verify namespace was taken from environment variable
        assert_eq!(args.namespace, "test-namespace");
    }
}
