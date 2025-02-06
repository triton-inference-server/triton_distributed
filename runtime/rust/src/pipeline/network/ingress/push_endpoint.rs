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
use anyhow::Result;
use async_nats::service::endpoint::Endpoint;
use derive_builder::Builder;
use tokio_util::sync::CancellationToken;
use tracing as log;

#[derive(Builder)]
pub struct PushEndpoint {
    pub service_handler: Arc<dyn PushWorkHandler>,
    pub cancellation_token: CancellationToken,
}

/// version of crate
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

impl PushEndpoint {
    pub fn builder() -> PushEndpointBuilder {
        PushEndpointBuilder::default()
    }

    pub async fn start(self, endpoint: Endpoint) -> Result<()> {
        let mut endpoint = endpoint;

        loop {
            let req = tokio::select! {
                biased;

                // await on service request
                req = endpoint.next() => {
                    req
                }

                // process shutdown
                _ = self.cancellation_token.cancelled() => {
                    // log::trace!(worker_id, "Shutting down service {}", self.endpoint.name);
                    if let Err(e) = endpoint.stop().await {
                        log::warn!("Failed to stop NATS service: {:?}", e);
                    }
                    break;
                }
            };

            if let Some(req) = req {
                let response = "".to_string();
                if let Err(e) = req.respond(Ok(response.into())).await {
                    log::warn!("Failed to respond to request; this may indicate the request has shutdown: {:?}", e);
                }

                let ingress = self.service_handler.clone();
                let worker_id = "".to_string();
                tokio::spawn(async move {
                    log::trace!(worker_id, "handling new request");
                    let result = ingress.handle_payload(req.message.payload).await;
                    log::trace!(worker_id, "request handled: {:?}", result);
                });
            } else {
                break;
            }
        }

        Ok(())
    }
}
