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

use std::path::Path;
use std::sync::Arc;

use triton_distributed_runtime::pipeline::error as pipeline_error;
use triton_distributed_runtime::CancellationToken;

use crate::backend::ExecutionContext;

mod worker;

mod engine;
use engine::VllmEngine;

mod subprocess;
pub use subprocess::run_subprocess;

pub async fn make_engine(
    cancel_token: CancellationToken,
    // Where to find the tokenzier, and config.json
    card_path: &Path,
    // Full path to the model, either a GGUF file or an HF repo dir
    model_path: &Path,
    // Unique string to name zmq sockets
    sock_code: &str,
) -> pipeline_error::Result<(ExecutionContext, tokio::task::JoinHandle<()>)> {
    let mut engine = VllmEngine::new(cancel_token, sock_code, card_path, model_path).await?;
    let vllm_process = engine.take_vllm_worker_handle();
    let engine: ExecutionContext = Arc::new(engine);
    Ok((engine, vllm_process))
}
