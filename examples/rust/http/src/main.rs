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
use std::env;

use triton_distributed::{logging, DistributedRuntime, Result, Runtime, Worker};
use triton_llm::http::service::{
    discovery::{model_watcher, ModelWatchState},
    service_v2::HttpService,
};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Port number for the HTTP service
    #[arg(short, long, default_value = "8080", env = "HTTP_SERVICE_PORT")]
    port: u16,

    /// Namespace for the distributed component
    #[arg(long, default_value = "public", env = "NAMESPACE")]
    namespace: String,

    /// Component name for the service
    #[arg(long, default_value = "http", env = "COMPONENT")]
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

    // create the http service and acquire the model manager
    let http_service = HttpService::builder().port(args.port).build()?;
    let manager = http_service.model_manager().clone();

    // todo - use the IntoComponent trait to register the component
    // todo - start a service
    // todo - we want the service to create an entry and register component definition
    // todo - the component definition should be the type of component and it's config
    // in this example we will have an HttpServiceComponentDefinition object which will be
    // written to etcd
    // the cli when operating on an `http` component will validate the namespace.component is
    // registered with HttpServiceComponentDefinition
    let component = distributed.namespace(&args.namespace)?.component(&args.component)?;

    let etcd_root = component.etcd_path();
    let etcd_path = format!("{}/models/chat/", etcd_root);

    let state = Arc::new(ModelWatchState {
        prefix: etcd_path.clone(),
        manager,
        drt: distributed.clone(),
    });

    let etcd_client = distributed.etcd_client();
    let models_watcher = etcd_client.kv_get_and_watch_prefix(etcd_path).await?;

    let (_prefix, _watcher, receiver) = models_watcher.dissolve();
    let _watcher_task = tokio::spawn(model_watcher(state, receiver));

    http_service.run(runtime.child_token()).await
}
