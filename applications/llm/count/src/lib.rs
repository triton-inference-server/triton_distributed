//! Library functions for the count application.

use prometheus::register_gauge_vec;
use serde::{Deserialize, Serialize};
use warp::Filter;

use triton_distributed_llm::kv_router::protocols::ForwardPassMetrics;
use triton_distributed_llm::kv_router::scheduler::Endpoint;
use triton_distributed_llm::kv_router::scoring::ProcessedEndpoints;

use triton_distributed_runtime::{
    distributed::Component, service::EndpointInfo, utils::Duration, Result,
};

/// Configuration for LLM worker load capacity metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMWorkerLoadCapacityConfig {
    pub component_name: String,
    pub endpoint_name: String,
}

// TODO: This is _really_ close to the async_nats::service::Stats object,
// but it's missing a few fields like "name", so use a temporary struct
// for easy deserialization. Ideally, this type already exists or can
// be exposed in the library somewhere.
/// Stats structure returned from NATS service API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatsWithData {
    // Standard NATS Service API fields
    pub average_processing_time: f64,
    pub last_error: String,
    pub num_errors: u64,
    pub num_requests: u64,
    pub processing_time: u64,
    pub queue_group: String,
    // Field containing custom stats handler data
    pub data: serde_json::Value,
}

/// Prometheus metrics server for exposing metrics
pub struct PrometheusMetricsServer {
    metrics: PrometheusMetrics,
}

impl PrometheusMetricsServer {
    /// Initialize the metrics server
    pub fn new() -> Result<Self> {
        Ok(Self {
            metrics: PrometheusMetrics::new()?,
        })
    }

    /// Start the metrics server on the specified port
    pub fn start(&mut self, port: u16) {
        let metrics_route = warp::path!("metrics").map(|| {
            use prometheus::Encoder;
            let encoder = prometheus::TextEncoder::new();
            let mut buffer = Vec::new();
            encoder.encode(&prometheus::gather(), &mut buffer).unwrap();
            String::from_utf8(buffer).unwrap()
        });

        // TODO: Use axum instead of warp for consistency and less dependencies
        let server = warp::serve(metrics_route).run(([0, 0, 0, 0], port));
        tokio::spawn(server);
        tracing::info!("Prometheus metrics server started on port {}", port);
    }

    /// Update metrics with current values
    pub fn update(&mut self, config: &LLMWorkerLoadCapacityConfig, processed: &ProcessedEndpoints) {
        self.metrics.update(config, processed);
    }
}

/// Prometheus metrics collection
pub struct PrometheusMetrics {
    kv_blocks_active: prometheus::GaugeVec,
    kv_blocks_total: prometheus::GaugeVec,
    requests_active: prometheus::GaugeVec,
    requests_total: prometheus::GaugeVec,
    load_avg: prometheus::GaugeVec,
    load_std: prometheus::GaugeVec,
}

impl PrometheusMetrics {
    /// Initialize all metrics
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

    /// Update metrics with current values
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

/// Collect endpoints from a component
pub async fn collect_endpoints(
    component: &Component,
    subject: &str,
    timeout: Duration,
) -> Result<Vec<EndpointInfo>> {
    // Collect stats from each backend
    let stream = component.scrape_stats(timeout).await?;

    // Filter the stats by the service subject
    let endpoints = stream
        .into_endpoints()
        .filter(|e| e.subject.starts_with(subject))
        .collect::<Vec<_>>();
    tracing::debug!("Endpoints: {endpoints:?}");

    if endpoints.is_empty() {
        tracing::warn!("No endpoints found matching subject {}", subject);
    }

    Ok(endpoints)
}

/// Extract metrics from endpoints
pub fn extract_metrics(endpoints: &[EndpointInfo]) -> Vec<ForwardPassMetrics> {
    let endpoint_data = endpoints.iter().map(|e| e.data.clone()).collect::<Vec<_>>();

    // Extract StatsWithData objects from endpoint services
    let stats: Vec<StatsWithData> = endpoint_data
        .iter()
        .filter_map(|e| {
            let metrics_data = e.as_ref()?;
            serde_json::from_value::<StatsWithData>(metrics_data.0.clone()).ok()
        })
        .collect();
    tracing::debug!("Stats: {stats:?}");

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

/// Create ProcessedEndpoints from metrics and endpoints
pub fn postprocess_metrics(
    metrics: &[ForwardPassMetrics],
    endpoints: &[EndpointInfo],
) -> ProcessedEndpoints {
    let processed_endpoints: Vec<Endpoint> = metrics
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

    ProcessedEndpoints::new(processed_endpoints)
}
