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

use std::path::PathBuf;

use triton_distributed_runtime::runtime::CancellationToken;
use triton_distributed_llm::types::openai::chat_completions::OpenAIChatCompletionsStreamingEngine;

mod input;
mod opt;
mod output;
pub use opt::{Input, Output};

/// Required options depend on the in and out choices
#[derive(clap::Parser, Debug, Clone)]
#[command(version, about, long_about = None)]
pub struct Flags {
    /// HTTP port. `in=http` only
    #[arg(long, default_value = "8080")]
    pub http_port: u16,

    /// The name of the model we are serving
    #[arg(long)]
    pub model_name: Option<String>,

    /// Full path to the model. This differs by engine:
    /// - mistralrs: File. GGUF.
    /// - echo_full: Omit the flag.
    #[arg(long)]
    pub model_path: Option<PathBuf>,
}

pub enum EngineConfig {
    /// A Full service engine does it's own tokenization and prompt formatting.
    StaticFull {
        service_name: String,
        engine: OpenAIChatCompletionsStreamingEngine,
    },
}

pub async fn run(
    in_opt: Input,
    out_opt: Output,
    flags: Flags,
    cancel_token: CancellationToken,
) -> anyhow::Result<()> {
    // Turn relative paths into absolute paths
    let model_path = flags.model_path.and_then(|p| p.canonicalize().ok());
    // Serve the model under the name provided, or the name of the GGUF file.
    let model_name = flags.model_name.or_else(||
            // "stem" means the filename without the extension.
            model_path.as_ref()
                .and_then(|p| p.file_stem())
                .map(|n| n.to_string_lossy().into_owned()));

    // Create the engine matching `out`
    let engine_config = match out_opt {
        Output::EchoFull => {
            let Some(model_name) = model_name else {
                anyhow::bail!(
                    "Pass --model-name or --model-path so we know which model to imitate"
                );
            };
            EngineConfig::StaticFull {
                service_name: model_name,
                engine: output::echo_full::make_engine_full(),
            }
        }
        #[cfg(feature = "mistralrs")]
        Output::MistralRs => {
            let Some(model_path) = model_path else {
                anyhow::bail!("out=mistralrs requires flag --model-path=<full-path-to-model-gguf>");
            };
            if !model_path.is_file() {
                anyhow::bail!("--model-path should refer to a GGUF file");
            }
            let Some(model_name) = model_name else {
                unreachable!("We checked model_path earlier, and set model_name from model_path");
            };
            EngineConfig::StaticFull {
                service_name: model_name,
                engine: triton_llm::engines::mistralrs::make_engine(&model_path).await?,
            }
        }
    };

    match in_opt {
        Input::Http => {
            crate::input::http::run(cancel_token.clone(), flags.http_port, engine_config).await?;
        }
        Input::Text => {
            crate::input::text::run(cancel_token.clone(), engine_config).await?;
        }
    }

    Ok(())
}
