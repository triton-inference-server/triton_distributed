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

use anyhow::Result;
use futures::stream::StreamExt;
use std::{sync::Arc, time::Duration};
use tokio_util::sync::CancellationToken;
use tracing as log;
use triton_distributed_runtime::{component::Component, DistributedRuntime};

pub mod indexer;
pub mod protocols;
pub mod publisher;
// [WIP] enable service_builder() through worker for metrics reporting
// pub mod worker;
mod scheduler;
mod scoring;

use crate::kv_router::{
    indexer::{KvIndexer, KvIndexerInterface, RouterEvent},
    protocols::KV_BLOCK_SIZE,
    scheduler::{Endpoint, KvScheduler, Service},
    scoring::ProcessedEndpoints,
};

// this should be discovered from the backend
pub const KV_EVENT_SUBJECT: &str = "kv_events";

pub struct KvRouter {
    // properties of request plane
    // maybe rolled up into the generic object or not
    service_name: String,

    cancellation_token: CancellationToken,

    #[allow(dead_code)]
    scheduler: KvScheduler,

    indexer: KvIndexer,
}

impl KvRouter {
    pub async fn from_runtime(
        runtime: DistributedRuntime,
        backend: Component,
    ) -> Result<Arc<Self>> {
        let nats_client = runtime.nats_client();
        let service_name = backend.service_name();
        let kv_subject = backend.event_subject(KV_EVENT_SUBJECT);
        log::info!("Component Service Name {}", service_name);
        log::info!("KV Subject {}", kv_subject);
        Self::new(nats_client, service_name, kv_subject).await
    }

    pub async fn new(
        nats_client: triton_distributed_runtime::transports::nats::Client,
        service_name: String,
        kv_subject: String,
    ) -> Result<Arc<Self>> {
        let cancellation_token = CancellationToken::new();
        let (ep_tx, ep_rx) = tokio::sync::mpsc::channel(128);

        tokio::spawn(collect_endpoints(
            nats_client.clone(),
            service_name.clone(),
            ep_tx,
            cancellation_token.clone(),
        ));

        let indexer = KvIndexer::new(cancellation_token.clone());
        let scheduler = KvScheduler::start(ep_rx).await?;

        log::debug!("subscribing to kv events: {}", kv_subject);
        let mut kv_events_rx = nats_client.client().subscribe(kv_subject).await?;
        let kv_events_tx = indexer.event_sender();

        tokio::spawn(async move {
            while let Some(event) = kv_events_rx.next().await {
                let event: RouterEvent = serde_json::from_slice(&event.payload).unwrap();
                log::debug!("received kv event: {:?}", event);
                if let Err(e) = kv_events_tx.send(event).await {
                    log::trace!("failed to send kv event to indexer; shutting down: {:?}", e);
                }
            }
        });

        Ok(Arc::new(Self {
            service_name,
            cancellation_token,
            scheduler,
            indexer,
        }))
    }

    pub fn cancellation_token(&self) -> CancellationToken {
        self.cancellation_token.clone()
    }

    pub fn service_name(&self) -> &str {
        &self.service_name
    }

    // [TODO] indexer needs to take 'lora_id' as parameter
    pub async fn schedule(&self, token_ids: &Vec<u32>, _lora_id: u64) -> Result<String> {
        // Extracting part of the code in KvRouter::generate() for only
        // the decision making part, routing is done by the caller
        let isl_tokens = token_ids.len();
        let overlap_scores = self
            .indexer
            .find_matches_for_request(token_ids.as_slice())
            .await?;
        log::debug!("KV router overlap_scores: {:?}", overlap_scores);
        // [FIXME] Python binding results in "endpoint subscriber shutdown" error,
        // need to investigate whether it happens in pure rust as well and then
        // root cause it. Before that, not doing intelligent scheduling for rapid
        // development..
        // [FIXME] also need to fix that scheduler returns worker subject which is not
        // the same as worker id (uuid). Seems like it adds additional annotation on top of uuid.
        // Need to double check
        // 'worker_subject' should be the same as worker id used for direct routing
        // let worker_subject = self.scheduler.schedule(overlap_scores, isl_tokens).await?;
        let mut selected_worker_subject = Option::<String>::None;
        for (worker_subject, overlap_score) in &overlap_scores.scores {
            if ((*overlap_score as usize * KV_BLOCK_SIZE) as f64 / isl_tokens as f64) >= 0.5 {
                selected_worker_subject = Some(worker_subject.to_string());
            }
        }
        match selected_worker_subject {
            None => Err(anyhow::anyhow!("No worker found")),
            Some(worker_subject) => Ok(worker_subject),
        }
    }
}

async fn collect_endpoints(
    nats_client: triton_distributed_runtime::transports::nats::Client,
    service_name: String,
    ep_tx: tokio::sync::mpsc::Sender<ProcessedEndpoints>,
    cancel: CancellationToken,
) {
    loop {
        tokio::select! {
            _ = cancel.cancelled() => {
                log::debug!("cancellation token triggered");
                break;
            }
            _ = tokio::time::sleep(Duration::from_secs(1)) => {
                log::trace!("collecting endpoints for service: {}", service_name);
            }
        }

        let values = nats_client
            .get_endpoints(&service_name, Duration::from_secs(1))
            .await
            .unwrap();

        // [FIXME] Endpoint is parsed from nats stats handler which may not include 'data' field
        // if the service hasn't registered the handler.
        // Another option is to make sure the router is configured properly that
        // it listens to the right subject (where other publisher has stats).
        let services: Vec<Service> = values
            .into_iter()
            .filter(|v| !v.is_empty())
            .map(|v| {
                let value: serde_json::Value = serde_json::from_slice(&v).unwrap();
                log::trace!("service value: {:?}", value);
                serde_json::from_slice(&v).unwrap()
            })
            .collect();

        let endpoints: Vec<Endpoint> = services.into_iter().flat_map(|s| s.endpoints).collect();

        log::trace!(
            "found {} endpoints for service: {}",
            endpoints.len(),
            service_name
        );

        let processed = ProcessedEndpoints::new(endpoints);

        // process endpoints into
        if ep_tx.send(processed).await.is_err() {
            log::trace!("failed to send processed endpoints; shutting down");
            break;
        }
    }
}
