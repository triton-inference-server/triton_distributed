/*
 * Copyright 2024-2025 NVIDIA CORPORATION & AFFILIATES
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License. You may obtain a copy of
 * the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

// TODO - refactor this entire module
//
// we want to carry forward the concept of live vs ready for the components
// we will want to associate the components cancellation token with the
// component's "service state"

use crate::{log, transports::nats, Result};

use async_nats::Message;
use async_stream::try_stream;
use bytes::Bytes;
use derive_getters::Dissolve;
use futures::stream::StreamExt;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::time::Duration;

pub struct ServiceClient {
    nats_client: nats::Client,
}

impl ServiceClient {
    #[allow(dead_code)]
    pub(crate) fn new(nats_client: nats::Client) -> Self {
        ServiceClient { nats_client }
    }
}

pub struct ServiceSet {
    services: Vec<ServiceInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceInfo {
    pub name: String,
    pub id: String,
    pub version: String,
    pub started: String,
    pub endpoints: Vec<EndpointInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Dissolve)]
pub struct EndpointInfo {
    pub name: String,
    pub subject: String,

    #[serde(flatten)]
    pub data: Metrics,
}

#[derive(Debug, Clone, Serialize, Deserialize, Dissolve)]
pub struct Metrics(pub serde_json::Value);

impl Metrics {
    pub fn decode<T: DeserializeOwned>(self) -> Result<T> {
        serde_json::from_value(self.0).map_err(Into::into)
    }
}

impl ServiceClient {
    pub async fn unary(
        &self,
        subject: impl Into<String>,
        payload: impl Into<Bytes>,
    ) -> Result<Message> {
        let response = self
            .nats_client
            .client()
            .request(subject.into(), payload.into())
            .await?;
        Ok(response)
    }

    pub async fn collect_services(&self, service_name: &str) -> Result<ServiceSet> {
        let mut sub = self.nats_client.service_subscriber(service_name).await?;
        let deadline = tokio::time::Instant::now() + Duration::from_secs(1);

        let services: Vec<Result<ServiceInfo>> = try_stream! {
            while let Ok(Some(message)) = tokio::time::timeout_at(deadline, sub.next()).await {
                if message.payload.is_empty() {
                    continue;
                }
                let service = serde_json::from_slice::<ServiceInfo>(&message.payload)?;
                log::trace!("service: {:?}", service);
                yield service;
            }
        }
        .collect()
        .await;

        // split ok and error results
        let (ok, err): (Vec<_>, Vec<_>) = services.into_iter().partition(Result::is_ok);

        if !err.is_empty() {
            log::error!("failed to collect services: {:?}", err);
        }

        Ok(ServiceSet {
            services: ok.into_iter().map(Result::unwrap).collect(),
        })
    }
}

impl ServiceSet {
    pub fn into_endpoints(self) -> impl Iterator<Item = EndpointInfo> {
        self.services
            .into_iter()
            .flat_map(|s| s.endpoints.into_iter())
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_service_set() {
        let services = vec![
            ServiceInfo {
                name: "service1".to_string(),
                id: "1".to_string(),
                version: "1.0".to_string(),
                started: "2021-01-01".to_string(),
                endpoints: vec![
                    EndpointInfo {
                        name: "endpoint1".to_string(),
                        subject: "subject1".to_string(),
                        data: Metrics(serde_json::json!({"key": "value1"})),
                    },
                    EndpointInfo {
                        name: "endpoint2-foo".to_string(),
                        subject: "subject2".to_string(),
                        data: Metrics(serde_json::json!({"key": "value1"})),
                    },
                ],
            },
            ServiceInfo {
                name: "service1".to_string(),
                id: "2".to_string(),
                version: "1.0".to_string(),
                started: "2021-01-01".to_string(),
                endpoints: vec![
                    EndpointInfo {
                        name: "endpoint1".to_string(),
                        subject: "subject1".to_string(),
                        data: Metrics(serde_json::json!({"key": "value1"})),
                    },
                    EndpointInfo {
                        name: "endpoint2-bar".to_string(),
                        subject: "subject2".to_string(),
                        data: Metrics(serde_json::json!({"key": "value2"})),
                    },
                ],
            },
        ];

        let service_set = ServiceSet { services };

        let endpoints: Vec<_> = service_set
            .into_endpoints()
            .filter(|e| e.name.starts_with("endpoint2"))
            .collect();

        assert_eq!(endpoints.len(), 2);
    }
}
