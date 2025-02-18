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

// TODO - refactor this entire module
//
// we want to carry forward the concept of live vs ready for the components
// we will want to associate the components cancellation token with the
// component's "service state"

use crate::{transports::nats, utils::stream, Result};

use async_nats::Message;
use async_stream::try_stream;
use bytes::Bytes;
use derive_getters::Dissolve;
use futures::stream::{StreamExt, TryStreamExt};
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
    pub data: Option<Metrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Dissolve)]
pub struct Metrics(pub serde_json::Value);

impl Metrics {
    pub fn decode<T: for<'de> Deserialize<'de>>(self) -> Result<T> {
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

    pub async fn collect_services(
        &self,
        service_name: &str,
        duration: Duration,
    ) -> Result<ServiceSet> {
        let sub = self.nats_client.scrape_service(service_name).await?;
        if duration.is_zero() {
            tracing::warn!("collect_services: duration is zero");
        }
        if duration > Duration::from_secs(10) {
            tracing::warn!("collect_services: duration is greater than 10 seconds");
        }
        let deadline = tokio::time::Instant::now() + duration;

        let services = stream::until_deadline(sub, deadline)
            .map(|message| serde_json::from_slice::<ServiceInfo>(&message.payload))
            .filter_map(|info| async move {
                match info {
                    Ok(info) => Some(info),
                    Err(e) => {
                        log::debug!("error decoding service info: {:?}", e);
                        None
                    }
                }
            })
            .collect()
            .await;

        Ok(ServiceSet { services })
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
                        data: Some(Metrics(serde_json::json!({"key": "value1"}))),
                    },
                    EndpointInfo {
                        name: "endpoint2-foo".to_string(),
                        subject: "subject2".to_string(),
                        data: Some(Metrics(serde_json::json!({"key": "value1"}))),
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
                        data: Some(Metrics(serde_json::json!({"key": "value1"}))),
                    },
                    EndpointInfo {
                        name: "endpoint2-bar".to_string(),
                        subject: "subject2".to_string(),
                        data: Some(Metrics(serde_json::json!({"key": "value2"}))),
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
