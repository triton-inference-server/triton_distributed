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

use triton_distributed_llm::{
    backend::ExecutionContext,
    model_card::model::ModelDeploymentCard,
    types::{
        openai::chat_completions::{
            ChatCompletionRequest, ChatCompletionResponseDelta,
            OpenAIChatCompletionsStreamingEngine,
        },
        Annotated,
    },
};
use triton_distributed_runtime::{component::Client, DistributedRuntime};

mod input;
mod opt;
mod output;
pub use opt::{Input, Output};

/// How we identify a namespace/component/endpoint URL.
/// Technically the '://' is not part of the scheme but it eliminates several string
/// concatenations.
const ENDPOINT_SCHEME: &str = "tdr://";

/// Required options depend on the in and out choices
#[derive(clap::Parser, Debug, Clone)]
#[command(version, about, long_about = None)]
pub struct Flags {
    /// Full path to the model, which can be either a GGUF file or a checked out HF repository.
    /// For the `echo_full` engine omit the flag.
    #[arg(index = 1)]
    pub model_path_pos: Option<PathBuf>,

    // `--model-path`. The one above is `tio <positional-model-path>`
    #[arg(long = "model-path")]
    pub model_path_flag: Option<PathBuf>,

    /// HTTP port. `in=http` only
    #[arg(long, default_value = "8080")]
    pub http_port: u16,

    /// The name of the model we are serving
    #[arg(long)]
    pub model_name: Option<String>,
}

pub enum EngineConfig {
    /// An remote networked engine we don't know about yet
    /// We don't have the pre-processor yet so this is only text requests. Type will change later.
    Dynamic(Client<ChatCompletionRequest, Annotated<ChatCompletionResponseDelta>>),

    /// A Full service engine does it's own tokenization and prompt formatting.
    StaticFull {
        service_name: String,
        engine: OpenAIChatCompletionsStreamingEngine,
    },

    /// A core engine expects to be wrapped with pre/post processors that handle tokenization.
    StaticCore {
        service_name: String,
        engine: ExecutionContext,
        card: Box<ModelDeploymentCard>,
    },
}

pub async fn run(
    runtime: triton_distributed_runtime::Runtime,
    in_opt: Input,
    out_opt: Output,
    flags: Flags,
) -> anyhow::Result<()> {
    let cancel_token = runtime.primary_token();

    // Turn relative paths into absolute paths
    let model_path = flags
        .model_path_pos
        .or(flags.model_path_flag)
        .and_then(|p| p.canonicalize().ok());
    // Serve the model under the name provided, or the name of the GGUF file.
    let model_name = flags.model_name.or_else(|| {
        model_path
            .as_ref()
            .and_then(|p| p.iter().last())
            .map(|n| n.to_string_lossy().into_owned())
    });
    // If model path is a directory we can build a model deployment card from it
    let maybe_card = match &model_path {
        Some(model_path) if model_path.is_dir() => {
            ModelDeploymentCard::from_local_path(model_path, model_name.as_deref())
                .await
                .ok()
        }
        Some(_) | None => None,
    };

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
        Output::EchoCore => {
            let Some(mut card) = maybe_card.clone() else {
                anyhow::bail!(
                    "out=echo_core need to find the tokenizer. Pass flag --model-path <path>"
                );
            };
            card.requires_preprocessing = true;
            EngineConfig::StaticCore {
                service_name: card.service_name.clone(),
                engine: output::echo_core::make_engine_core(),
                card: Box::new(card),
            }
        }
        Output::Endpoint(path) => {
            let elements: Vec<&str> = path.split('/').collect();
            if elements.len() != 3 {
                anyhow::bail!("An endpoint URL must have format {ENDPOINT_SCHEME}namespace/component/endpoint");
            }
            // This will attempt to connect to NATS and etcd
            let distributed_runtime = DistributedRuntime::from_settings(runtime.clone()).await?;

            let client = distributed_runtime
                .namespace(elements[0])?
                .component(elements[1])?
                .endpoint(elements[2])
                .client::<ChatCompletionRequest, Annotated<ChatCompletionResponseDelta>>()
                .await?;

            tracing::info!("Waiting for remote {}...", client.path());
            tokio::select! {
                _ = cancel_token.cancelled() => {
                    return Ok(());
                }
                r = client.wait_for_endpoints() => {
                    r?;
                }
            }

            EngineConfig::Dynamic(client)
        }
        #[cfg(feature = "mistralrs")]
        Output::MistralRs => {
            let Some(model_path) = model_path else {
                anyhow::bail!("out=mistralrs requires flag --model-path=<full-path-to-model-gguf>");
            };
            let Some(model_name) = model_name else {
                unreachable!("We checked model_path earlier, and set model_name from model_path");
            };
            EngineConfig::StaticFull {
                service_name: model_name,
                engine: triton_distributed_llm::engines::mistralrs::make_engine(&model_path)
                    .await?,
            }
        }
    };

    match in_opt {
        Input::Http => {
            crate::input::http::run(runtime.clone(), flags.http_port, engine_config).await?;
        }
        Input::Text => {
            crate::input::text::run(cancel_token.clone(), engine_config).await?;
        }
        Input::Endpoint(path) => {
            crate::input::endpoint::run(runtime.clone(), path, engine_config).await?;
        }
    }

    Ok(())
}
