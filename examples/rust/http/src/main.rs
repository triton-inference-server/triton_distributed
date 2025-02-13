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

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc::Receiver, Mutex};
use triton_distributed::{
    logging,
    pipeline::{
        async_trait, network::Ingress, AsyncEngine, AsyncEngineContextProvider, Error, ManyOut,
        ResponseStream, SingleIn,
    },
    protocols::annotated::Annotated,
    raise,
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
    logging::init();
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

#[derive(Debug, Clone, Serialize, Deserialize)]
enum ApiType {
    #[serde(rename = "openai_chat")]
    OpenAIChat,

    #[serde(rename = "openai_cmpl")]
    OpenAICmpl,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ModelRegistryRequest {
    AddModel {
        model_entry: ModelEntry,
        api_type: ApiType,
    },
    RemoveModel {
        model_name: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModelRegistryResponse {}

struct ModelState {
    models: HashMap<String, ModelEntry>,
    instances: HashMap<String, HashSet<i64>>,
}

/// [ModelDiscovery] is a struct that contains the information for the HTTP service to discover models
/// from the etcd cluster.
///
/// When started, this will create a task that watches for llm::http::ModelEntry
struct ModelRegistry {
    models: Mutex<HashMap<String, ModelEntry>>,
}

impl ModelRegistry {
    pub async fn handle_request(&self, request: ModelRegistryRequest) -> Result<()> {
        match request {
            ModelRegistryRequest::AddModel {
                model_entry,
                api_type,
            } => self.add_model(model_entry, api_type).await,
            ModelRegistryRequest::RemoveModel { model_name } => self.remove_model(model_name).await,
        }
    }

    async fn add_model(&self, model_entry: ModelEntry, api_type: ApiType) -> Result<()> {
        let mut models = self.models.lock().await;

        // if model_name exists, the endpoint much match
        if let Some(existing_model) = models.get(&model_entry.name) {
            if existing_model.endpoint != model_entry.endpoint {
                raise!("Endpoint mismatch for model: {}", model_entry.name);
            }

            // for each api type, write an entry to the etcd cluster
            self.write_etcd_entry(&model_entry, &api_type).await
        } else {
            self.write_etcd_entry(&model_entry, &api_type).await?;
            models.insert(model_entry.name.clone(), model_entry);
            Ok(())
        }
    }

    async fn remove_model(&self, model_name: String) -> Result<()> {
        unimplemented!()
    }

    async fn write_etcd_entry(&self, model_entry: &ModelEntry, api_type: &ApiType) -> Result<()> {
        unimplemented!()
    }
}

#[async_trait]
impl AsyncEngine<SingleIn<ModelRegistryRequest>, ManyOut<Annotated<ModelRegistryResponse>>, Error>
    for ModelRegistry
{
    async fn generate(
        &self,
        input: SingleIn<ModelRegistryRequest>,
    ) -> Result<ManyOut<Annotated<ModelRegistryResponse>>> {
        // check to see if the request is a valid request
        // the model entry should match
        let (request, ctx) = input.into_parts();

        // if the request is valid, add the model entry to the map

        let stream = stream::iter(chars);

        Ok(ResponseStream::new(Box::pin(stream), ctx.context()))
    }
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
