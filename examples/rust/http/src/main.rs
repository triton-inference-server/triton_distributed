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
use clap::Parser;

use triton_distributed::{logging, DistributedRuntime, Result, Runtime, Worker};
use triton_llm::http::service::{
    discovery::{model_watcher, ModelWatchState},
    service_v2::HttpService,
};
use triton_llm::model_type::ModelType;  // Import the shared ModelType

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Host for the HTTP service
    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    /// Port number for the HTTP service
    #[arg(short, long, default_value = "8080")]
    port: u16,

    /// Namespace for the distributed component
    #[arg(long, default_value = "public")]
    namespace: String,

    /// Component name for the service
    #[arg(long, default_value = "http")]
    component: String,
}

fn main() -> Result<()> {
    logging::init();
    let worker = Worker::from_settings()?;
    worker.execute(app)
}

async fn app(runtime: Runtime) -> Result<()> {
    let distributed = DistributedRuntime::from_settings(runtime.clone()).await?;
    let args = Args::parse();

    let http_service = HttpService::builder()
        .port(args.port)
        .host(args.host)
        .build()?;
    let manager = http_service.model_manager().clone();

    let component = distributed.namespace(&args.namespace)?.component(&args.component)?;
    let etcd_root = component.etcd_path();

    // Create watchers for all model types
    let mut watcher_tasks = Vec::new();

    for model_type in ModelType::all() {
        let etcd_path = format!("{}/models/{}/", etcd_root, model_type.as_str());
        let etcd_path_pop = format!("{}/models/", etcd_root);

        let state = Arc::new(ModelWatchState {
            prefix_to_name: etcd_path.clone(),
            prefix_to_type: etcd_path_pop.clone(),
            manager: manager.clone(),
            drt: distributed.clone(),
        });

        let etcd_client = distributed.etcd_client();
        let models_watcher: triton_distributed::transports::etcd::PrefixWatcher = etcd_client.kv_get_and_watch_prefix(etcd_path).await?;

        let (_prefix, _watcher, receiver) = models_watcher.dissolve();
        let watcher_task = tokio::spawn(model_watcher(state, receiver));
        watcher_tasks.push(watcher_task);
    }

    // Run the service
    http_service.run(runtime.child_token()).await
}
