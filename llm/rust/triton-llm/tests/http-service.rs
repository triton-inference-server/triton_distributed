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

use anyhow::Error;
use async_stream::stream;
use prometheus::{proto::MetricType, Registry};
use reqwest::StatusCode;
use std::sync::Arc;
use triton_distributed::{
    pipeline::{
        async_trait, AsyncEngine, AsyncEngineContextProvider, ManyOut, ResponseStream, SingleIn,
    },
    CancellationToken,
};
use triton_llm::http::service::{
    error::HttpError,
    metrics::{Endpoint, RequestType, Status},
    service_v2::HttpService,
    Metrics,
};
use triton_llm::protocols::{
    openai::{
        chat_completions::{ChatCompletionRequest, ChatCompletionResponseDelta},
        completions::{CompletionRequest, CompletionResponse},
    },
    Annotated,
};

struct CounterEngine {}

#[async_trait]
impl
    AsyncEngine<
        SingleIn<ChatCompletionRequest>,
        ManyOut<Annotated<ChatCompletionResponseDelta>>,
        Error,
    > for CounterEngine
{
    async fn generate(
        &self,
        request: SingleIn<ChatCompletionRequest>,
    ) -> Result<ManyOut<Annotated<ChatCompletionResponseDelta>>, Error> {
        let (request, context) = request.transfer(());
        let ctx = context.context();

        let max_tokens = request.max_tokens.unwrap_or(0) as u64;

        // let generator = ChatCompletionResponseDelta::generator(request.model.clone());
        let generator = request.response_generator();

        let stream = stream! {
            tokio::time::sleep(std::time::Duration::from_millis(max_tokens)).await;
            for i in 0..10 {
                let choice = generator.create_choice(i as u64,Some(format!("choice {i}")), None, None);
                yield Annotated::from_data(choice);
            }
        };

        Ok(ResponseStream::new(Box::pin(stream), ctx))
    }
}

struct AlwaysFailEngine {}

#[async_trait]
impl
    AsyncEngine<
        SingleIn<ChatCompletionRequest>,
        ManyOut<Annotated<ChatCompletionResponseDelta>>,
        Error,
    > for AlwaysFailEngine
{
    async fn generate(
        &self,
        _request: SingleIn<ChatCompletionRequest>,
    ) -> Result<ManyOut<Annotated<ChatCompletionResponseDelta>>, Error> {
        Err(HttpError {
            code: 403,
            message: "Always fail".to_string(),
        })?
    }
}

#[async_trait]
impl AsyncEngine<SingleIn<CompletionRequest>, ManyOut<Annotated<CompletionResponse>>, Error>
    for AlwaysFailEngine
{
    async fn generate(
        &self,
        _request: SingleIn<CompletionRequest>,
    ) -> Result<ManyOut<Annotated<CompletionResponse>>, Error> {
        Err(HttpError {
            code: 401,
            message: "Always fail".to_string(),
        })?
    }
}

fn compare_counter(
    metrics: Arc<Metrics>,
    model: &str,
    endpoint: &Endpoint,
    request_type: &RequestType,
    status: &Status,
    expected: u64,
) {
    assert_eq!(
        metrics.get_request_counter(model, endpoint, request_type, status),
        expected,
        "model: {}, endpoint: {:?}, request_type: {:?}, status: {:?}",
        model,
        endpoint.as_str(),
        request_type.as_str(),
        status.as_str()
    );
}

fn compute_index(endpoint: &Endpoint, request_type: &RequestType, status: &Status) -> usize {
    let endpoint = match endpoint {
        Endpoint::Completions => 0,
        Endpoint::ChatCompletions => 1,
    };

    let request_type = match request_type {
        RequestType::Unary => 0,
        RequestType::Stream => 1,
    };

    let status = match status {
        Status::Success => 0,
        Status::Error => 1,
    };

    endpoint * 4 + request_type * 2 + status
}

fn compare_counters(metrics: Arc<Metrics>, model: &str, expected: &[u64; 8]) {
    for endpoint in &[Endpoint::Completions, Endpoint::ChatCompletions] {
        for request_type in &[RequestType::Unary, RequestType::Stream] {
            for status in &[Status::Success, Status::Error] {
                let index = compute_index(endpoint, request_type, status);
                compare_counter(
                    metrics.clone(),
                    model,
                    endpoint,
                    request_type,
                    status,
                    expected[index],
                );
            }
        }
    }
}

fn inc_counter(
    endpoint: Endpoint,
    request_type: RequestType,
    status: Status,
    expected: &mut [u64; 8],
) {
    let index = compute_index(&endpoint, &request_type, &status);
    expected[index] += 1;
}

#[tokio::test]
async fn test_http_service() {
    let mut builder = HttpService::builder().port(8989);

    let service = builder.build().unwrap();
    let manager = service.model_manager().clone();

    let token = CancellationToken::new();
    let cancel_token = token.clone();
    let task = tokio::spawn(async move { service.run(token.clone()).await });

    let registry = Registry::new();

    let counter = Arc::new(CounterEngine {});
    let result = manager.add_chat_completions_model("foo", counter);
    assert!(result.is_ok());

    let failure = Arc::new(AlwaysFailEngine {});
    let result = manager.add_chat_completions_model("bar", failure.clone());
    assert!(result.is_ok());

    let result = manager.add_completions_model("bar", failure);
    assert!(result.is_ok());

    let metrics = manager.metrics();
    metrics.register(&registry).unwrap();

    let mut foo_counters = [0u64; 8];
    let mut bar_counters = [0u64; 8];

    compare_counters(metrics.clone(), "foo", &foo_counters);
    compare_counters(metrics.clone(), "bar", &bar_counters);

    let client = reqwest::Client::new();

    let mut request = ChatCompletionRequest::builder()
        .model("foo")
        .add_user_message("hi")
        .build()
        .unwrap();

    // ==== ChatCompletions / Stream / Success ====
    request.stream = Some(true);
    request.max_tokens = Some(3000);

    let response = client
        .post("http://localhost:8989/v1/chat/completions")
        .json(&request)
        .send()
        .await
        .unwrap();

    assert!(response.status().is_success(), "{:?}", response);

    tokio::time::sleep(tokio::time::Duration::from_millis(1000)).await;
    assert_eq!(metrics.get_inflight_count("foo"), 1);

    // process byte stream
    let _ = response.bytes().await.unwrap();

    inc_counter(
        Endpoint::ChatCompletions,
        RequestType::Stream,
        Status::Success,
        &mut foo_counters,
    );
    compare_counters(metrics.clone(), "foo", &foo_counters);
    compare_counters(metrics.clone(), "bar", &bar_counters);

    // check registry and look or the request duration histogram
    let families = registry.gather();
    let histogram_metric_family = families
        .into_iter()
        .find(|m| m.get_name() == "nv_llm_http_service_request_duration_seconds")
        .expect("Histogram metric not found");

    assert_eq!(
        histogram_metric_family.get_field_type(),
        MetricType::HISTOGRAM
    );

    let histogram_metric = histogram_metric_family.get_metric();

    assert_eq!(histogram_metric.len(), 1); // We have one metric with label model

    let metric = &histogram_metric[0];
    let histogram = metric.get_histogram();

    let buckets = histogram.get_bucket();

    let mut found = false;

    for bucket in buckets {
        let upper_bound = bucket.get_upper_bound();
        let cumulative_count = bucket.get_cumulative_count();

        println!(
            "Bucket upper bound: {}, count: {}",
            upper_bound, cumulative_count
        );

        // Since our observation is 2.5, it should fall into the bucket with upper bound 4.0
        if upper_bound >= 4.0 {
            assert_eq!(
                cumulative_count, 1,
                "Observation should be counted in the 4.0 bucket"
            );
            found = true;
        } else {
            assert_eq!(
                cumulative_count, 0,
                "No observations should be in this bucket"
            );
        }
    }

    assert!(found, "The expected bucket was not found");
    // ==== ChatCompletions / Stream / Success ====

    // ==== ChatCompletions / Unary / Success ====
    request.stream = Some(false);
    request.max_tokens = Some(0);

    let future = client
        .post("http://localhost:8989/v1/chat/completions")
        .json(&request)
        .send();

    let response = future.await.unwrap();

    assert!(response.status().is_success(), "{:?}", response);
    inc_counter(
        Endpoint::ChatCompletions,
        RequestType::Unary,
        Status::Success,
        &mut foo_counters,
    );
    compare_counters(metrics.clone(), "foo", &foo_counters);
    compare_counters(metrics.clone(), "bar", &bar_counters);
    // ==== ChatCompletions / Unary / Success ====

    // ==== ChatCompletions / Stream / Error ====
    request.model = "bar".to_string();
    request.max_tokens = Some(0);
    request.stream = Some(true);

    let response = client
        .post("http://localhost:8989/v1/chat/completions")
        .json(&request)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::FORBIDDEN);
    inc_counter(
        Endpoint::ChatCompletions,
        RequestType::Stream,
        Status::Error,
        &mut bar_counters,
    );
    compare_counters(metrics.clone(), "foo", &foo_counters);
    compare_counters(metrics.clone(), "bar", &bar_counters);
    // ==== ChatCompletions / Stream / Error ====

    // ==== ChatCompletions / Unary / Error ====
    request.stream = Some(false);

    let response = client
        .post("http://localhost:8989/v1/chat/completions")
        .json(&request)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::FORBIDDEN);
    inc_counter(
        Endpoint::ChatCompletions,
        RequestType::Unary,
        Status::Error,
        &mut bar_counters,
    );
    compare_counters(metrics.clone(), "foo", &foo_counters);
    compare_counters(metrics.clone(), "bar", &bar_counters);
    // ==== ChatCompletions / Unary / Error ====

    // ==== Completions / Unary / Error ====
    let mut request = CompletionRequest::builder()
        .model("bar")
        .prompt("hi")
        .build()
        .unwrap();

    let response = client
        .post("http://localhost:8989/v1/completions")
        .json(&request)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    inc_counter(
        Endpoint::Completions,
        RequestType::Unary,
        Status::Error,
        &mut bar_counters,
    );
    compare_counters(metrics.clone(), "foo", &foo_counters);
    compare_counters(metrics.clone(), "bar", &bar_counters);
    // ==== Completions / Unary / Error ====

    // ==== Completions / Stream / Error ====
    request.stream = Some(true);

    let response = client
        .post("http://localhost:8989/v1/completions")
        .json(&request)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    inc_counter(
        Endpoint::Completions,
        RequestType::Stream,
        Status::Error,
        &mut bar_counters,
    );
    compare_counters(metrics.clone(), "foo", &foo_counters);
    compare_counters(metrics.clone(), "bar", &bar_counters);
    // ==== Completions / Stream / Error ====

    // =========== Test Invalid Request ===========
    // send a completion request to a chat endpoint
    request.stream = Some(false);

    let response = client
        .post("http://localhost:8989/v1/chat/completions")
        .json(&request)
        .send()
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        StatusCode::UNPROCESSABLE_ENTITY,
        "{:?}",
        response
    );

    // =========== Query /metrics endpoint ===========
    let response = client
        .get("http://localhost:8989/metrics")
        .send()
        .await
        .unwrap();

    assert!(response.status().is_success(), "{:?}", response);
    println!("{}", response.text().await.unwrap());

    cancel_token.cancel();
    task.await.unwrap().unwrap();
}
