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

use hello_world::DEFAULT_NAMESPACE;
use triton_distributed::{
    self, protocols::annotated::Annotated, stream::StreamExt, DistributedRuntime, Result, Runtime,
};

#[triton_distributed::main]
async fn app(runtime: Runtime) -> Result<()> {
    env_logger::init();
    let distributed = DistributedRuntime::from_settings(runtime.clone()).await?;

    let client = distributed
        .namespace(DEFAULT_NAMESPACE)?
        .component("backend")?
        .endpoint("generate")
        .client::<String, Annotated<String>>()
        .await?;

    client.wait_for_endpoints().await?;

    let mut stream = client.random("hello world".to_string().into()).await?;

    while let Some(resp) = stream.next().await {
        println!("{:?}", resp);
    }

    runtime.shutdown();

    Ok(())
}
