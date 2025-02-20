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

use crate::kv_router::{indexer::RouterEvent, protocols::*, KV_EVENT_SUBJECT};
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing as log;
use triton_distributed::{component::Component, DistributedRuntime, Result};

pub struct KvEventPublisher {
    tx: mpsc::UnboundedSender<KvCacheEvent>,
}

impl KvEventPublisher {
    pub fn new(drt: DistributedRuntime, backend: Component, worker_id: i64) -> Result<Self> {
        let (tx, rx) = mpsc::unbounded_channel::<KvCacheEvent>();
        let p = KvEventPublisher { tx };

        start_publish_task(drt, backend, worker_id, rx);
        Ok(p)
    }

    pub fn publish(&self, event: KvCacheEvent) -> Result<(), mpsc::error::SendError<KvCacheEvent>> {
        log::debug!("Publish event: {:?}", event);
        self.tx.send(event)
    }
}

fn start_publish_task(
    drt: DistributedRuntime,
    backend: Component,
    worker_id: i64,
    mut rx: mpsc::UnboundedReceiver<KvCacheEvent>,
) {
    let client = drt.nats_client().client().clone();
    let kv_subject = backend.event_subject(KV_EVENT_SUBJECT);
    log::info!("Publishing KV Events to subject: {}", kv_subject);

    _ = drt.runtime().secondary().spawn(async move {
        while let Some(event) = rx.recv().await {
            let router_event = RouterEvent::new(worker_id, event);
            let data = serde_json::to_string(&router_event).unwrap();
            client
                .publish(kv_subject.to_string(), data.into())
                .await
                .unwrap();
        }
    });
}

pub struct KvMetricsPublisher {
    tx: tokio::sync::watch::Sender<Arc<ForwardPassMetrics>>,
    rx: tokio::sync::watch::Receiver<Arc<ForwardPassMetrics>>,
}

impl KvMetricsPublisher {
    pub fn new() -> Result<Self> {
        let (tx, rx) = tokio::sync::watch::channel(Arc::new(ForwardPassMetrics::default()));
        Ok(KvMetricsPublisher { tx, rx })
    }

    pub fn publish(
        &self,
        metrics: Arc<ForwardPassMetrics>,
    ) -> Result<(), tokio::sync::watch::error::SendError<Arc<ForwardPassMetrics>>> {
        log::debug!("Publish metrics: {:?}", metrics);
        self.tx.send(metrics)
    }

    pub async fn create_service(&self, component: Component) -> Result<()> {
        let mut metrics_rx = self.rx.clone();
        let _ = component
            .service_builder()
            .stats_handler(Some(Box::new(move |name, stats| {
                log::debug!("[IN worker?] Stats for service {}: {:?}", name, stats);
                let metrics = metrics_rx.borrow_and_update().clone();
                serde_json::to_value(&*metrics).unwrap()
            })))
            .create()
            .await?;
        Ok(())
    }
}
