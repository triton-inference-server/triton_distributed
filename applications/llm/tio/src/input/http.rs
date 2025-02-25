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

use std::sync::Arc;

use triton_distributed_llm::http::service::{discovery, service_v2};
use triton_distributed_runtime::{DistributedRuntime, Runtime};

use crate::EngineConfig;

/// Build and run an HTTP service
pub async fn run(
    runtime: Runtime,
    http_port: u16,
    engine_config: EngineConfig,
) -> anyhow::Result<()> {
    let http_service = service_v2::HttpService::builder()
        .port(http_port)
        .enable_chat_endpoints(true)
        .enable_cmpl_endpoints(true)
        .build()?;
    match engine_config {
        EngineConfig::Dynamic(client) => {
            let service_name = client.path();
            let distributed_runtime = DistributedRuntime::from_settings(runtime.clone()).await?;
            // Listen for models registering themselves in etcd, add them to HTTP service
            let state = Arc::new(discovery::ModelWatchState {
                prefix: service_name.clone(),
                manager: http_service.model_manager().clone(),
                drt: distributed_runtime.clone(),
            });
            let etcd_client = distributed_runtime.etcd_client();
            let models_watcher = etcd_client.kv_get_and_watch_prefix(service_name).await?;
            let (_prefix, _watcher, receiver) = models_watcher.dissolve();
            let _watcher_task = tokio::spawn(discovery::model_watcher(state, receiver));
        }
        EngineConfig::StaticFull {
            service_name,
            engine,
            ..
        } => {
            http_service
                .model_manager()
                .add_chat_completions_model(&service_name, engine)?;
        }
    }
    http_service.run(runtime.primary_token()).await
}
