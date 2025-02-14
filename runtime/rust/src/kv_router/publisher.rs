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

use crate::kv_router::{indexer::RouterEvent, protocols::KvCacheEvent, KV_EVENT_SUBJECT};
use crate::{component::Component, DistributedRuntime, Result};
use tokio::sync::mpsc;
use uuid::Uuid;
use tracing as log;
use tracing_subscriber::FmtSubscriber;
use std::fs::OpenOptions;
use std::io::Write;

pub struct KvPublisher {
    tx: mpsc::UnboundedSender<KvCacheEvent>,
}

impl KvPublisher {
    pub fn new(drt: DistributedRuntime, backend: Component, worker_id: Uuid) -> Result<Self> {

        let subscriber = FmtSubscriber::builder()
            // all spans/events with a level higher than TRACE (e.g, debug, info, warn, etc.)
            // will be written to stdout.
            .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
            // completes the builder.
            .finish();

        tracing::subscriber::set_global_default(subscriber)
        .expect("setting default subscriber failed");
        
        log::info!("Logging Started in Publisher");

        let (tx, rx) = mpsc::unbounded_channel::<KvCacheEvent>();
        let p: KvPublisher = KvPublisher { tx };

        start_publish_task(drt, backend, worker_id, rx);
        Ok(p)
    }

    pub fn publish(&self, event: KvCacheEvent) -> Result<(), mpsc::error::SendError<KvCacheEvent>> {
        log::info!("Attempting to publish event: {:?}", event);
        self.tx.send(event)
    }
}

fn start_publish_task(
    drt: DistributedRuntime,
    backend: Component,
    worker_id: Uuid,
    mut rx: mpsc::UnboundedReceiver<KvCacheEvent>,
) {
    let client = drt.nats_client().client().clone();
    // [FIXME] service name is for metrics polling?
    // let service_name = backend.service_name();
    let kv_subject: String = backend.event_subject(KV_EVENT_SUBJECT);
    log::info!("Publishing to subject: {}", kv_subject);
    _ = drt.runtime().secondary().spawn(async move {
        while let Some(event) = rx.recv().await {
            let router_event: RouterEvent = RouterEvent::new(worker_id, event);
            let data: String = serde_json::to_string(&router_event).unwrap();
            client
                .publish(kv_subject.to_string(), data.into())
                .await
                .unwrap();
        }
    });
}
