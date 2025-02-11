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

use std::collections::HashMap;

use tokio::sync::mpsc::Receiver;
use triton_distributed::{
    transports::etcd::{self, WatchEvent},
    DistributedRuntime, Result, Runtime, Worker,
};
use triton_llm::http::{
    service::{
        service_v2::{HttpService, HttpServiceConfig},
        ModelManager,
    },
    ModelEntry,
};

fn main() -> Result<()> {
    env_logger::init();
    let worker = Worker::from_settings()?;
    worker.execute(app)
}

async fn app(runtime: Runtime) -> Result<()> {
    let distributed = DistributedRuntime::from_settings(runtime.clone()).await?;

    // this pattern stinks - fix it
    let http_service = HttpService::builder().port(9992).build()?;

    let component = distributed.namespace("public-api")?.component("http")?;
    let etcd_root = component.etcd_path();
    let etcd_path = format!("{}/models/", etcd_root);

    let manager = http_service.model_manager().clone();
    let etcd_client = distributed.etcd_client();
    let models_watcher = etcd_client.kv_get_and_watch_prefix(etcd_path).await?;

    let (prefix, _watcher, receiver) = models_watcher.dissolve();
    let watcher_task = tokio::spawn(model_watcher(prefix, etcd_client, manager, receiver));

    http_service.run(runtime.child_token()).await
}

/// [ModelDiscovery] is a struct that contains the information for the HTTP service to discover models
/// from the etcd cluster.
///
/// When started, this will create a task that watches for llm::http::ModelEntry
struct ModelRegistry {
    models: HashMap<String, ModelEntry>,
}

async fn model_watcher(
    prefix: String,
    etcd_client: etcd::Client,
    manager: ModelManager,
    events_rx: Receiver<WatchEvent>,
) -> Result<()> {
    let mut events_rx = events_rx;
    let mut map: HashMap<String, HashMap<i64, ModelEntry>> = HashMap::new();
    while let Some(event) = events_rx.recv().await {
        match event {
            WatchEvent::Put(kv) => {
                let key = kv.key_str()?;
                let value = serde_json::from_slice::<ModelEntry>(&kv.value())?;

                // trim prefix from key to get the model name
                let model_name = key.trim_start_matches(&prefix);

                // check to see if the model name is already in the map
            }
            WatchEvent::Delete(kv) => {}
        }
    }

    Ok(())
}
