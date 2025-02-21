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

use triton_distributed_runtime::runtime::CancellationToken;
use triton_distributed_llm::http::service::service_v2;

use crate::EngineConfig;

/// Build and run an HTTP service
pub async fn run(
    cancel_token: CancellationToken,
    http_port: u16,
    engine_config: EngineConfig,
) -> anyhow::Result<()> {
    match engine_config {
        EngineConfig::StaticFull {
            service_name,
            engine,
            ..
        } => {
            let http_service = service_v2::HttpService::builder()
                .port(http_port)
                .enable_chat_endpoints(true)
                .enable_cmpl_endpoints(true)
                .build()?;
            http_service
                .model_manager()
                .add_chat_completions_model(&service_name, engine)?;
            http_service.run(cancel_token).await
        }
    }
}
