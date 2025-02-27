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
use std::str::FromStr;

use triton_distributed_llm::{
    backend::ExecutionContext,
    model_card::model::ModelDeploymentCard,
    types::{
        openai::chat_completions::{
            NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
            OpenAIChatCompletionsStreamingEngine,
        },
        Annotated,
    },
};
use triton_distributed_runtime::{component::Client, protocols::Endpoint, DistributedRuntime};

mod input;
#[cfg(feature = "sglang")]
mod net;
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

    /// sglang only
    ///
    /// How many GPUs to use at once, total across all nodes.
    /// This must divide by num_nodes, and each node must use the same number of GPUs.
    #[arg(long, default_value = "1", value_parser = clap::value_parser!(u32).range(1..256))]
    pub tensor_parallel_size: u32,

    /// sglang only
    ///
    /// Use GPUs from this ID upwards.
    /// If your machine has four GPUs but the first two (0 and 1) are in use,
    /// pass --base-gpu-id 2 to use the third GPU (and up, if tensor_parallel_size > 1)
    #[arg(long, default_value = "0", value_parser = clap::value_parser!(u32).range(0..256))]
    pub base_gpu_id: u32,

    /// sglang only
    ///
    /// How many nodes/hosts to use
    #[arg(long, default_value = "1", value_parser = clap::value_parser!(u32).range(1..256))]
    pub num_nodes: u32,

    /// sglang only
    ///
    /// This nodes' unique ID, running from 0 to num_nodes.
    #[arg(long, default_value = "0", value_parser = clap::value_parser!(u32).range(0..255))]
    pub node_rank: u32,

    /// sglang only
    ///
    /// The Torch Distributed init method address, in format <host>:<port>.
    /// It becomes "tcp://<host>:<port>" when given to torch.distributed.init_process_group.
    /// This expects to use the nccl backend (transparently to us here).
    /// All nodes must use the same dist_init_addr, which is node_rank == 0's address.
    #[arg(long)]
    pub dist_init_addr: Option<String>,

    /// Internal use only.
    /// Start the sglang Python sub-process.
    /// The params in the tuple are:
    /// - the fd of the write end of a pipe where sglang will signal that it's ready.
    /// - the node rank (0 for first host, 1 for second host, etc)
    /// - the workers' rank (globally unique)
    /// - the GPU to use (locally unique)
    #[arg(long)]
    #[clap(hide = true, value_parser = parse_sglang_flags)]
    pub internal_sglang_process: Option<SgLangFlags>,
}

pub enum EngineConfig {
    /// An remote networked engine we don't know about yet
    /// We don't have the pre-processor yet so this is only text requests. Type will change later.
    Dynamic(Client<NvCreateChatCompletionRequest, Annotated<NvCreateChatCompletionStreamResponse>>),

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

#[derive(Debug, Clone, Copy)]
pub struct SgLangFlags {
    pub pipe_fd: u32,
    pub tp_rank: u32,
    pub gpu_id: u32,
}
fn parse_sglang_flags(s: &str) -> Result<SgLangFlags, String> {
    let nums: Vec<u32> = s
        .split(',')
        .map(u32::from_str)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;

    if nums.len() != 3 {
        return Err("Need exactly 3 numbers".into());
    }

    Ok(SgLangFlags {
        pipe_fd: nums[0],
        tp_rank: nums[1],
        gpu_id: nums[2],
    })
}

pub async fn run(
    runtime: triton_distributed_runtime::Runtime,
    in_opt: Input,
    out_opt: Output,
    flags: Flags,
    #[allow(unused_variables)] zmq_socket_prefix: Option<String>,
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

    #[cfg(feature = "sglang")]
    let mut extra = None; // sglang sub-process

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
            let endpoint: Endpoint = path.parse()?;

            // This will attempt to connect to NATS and etcd
            let distributed_runtime = DistributedRuntime::from_settings(runtime.clone()).await?;

            let client = distributed_runtime
                .namespace(endpoint.namespace)?
                .component(endpoint.component)?
                .endpoint(endpoint.name)
                .client::<NvCreateChatCompletionRequest, Annotated<NvCreateChatCompletionStreamResponse>>()
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
        #[cfg(feature = "sglang")]
        Output::SgLang => {
            use triton_distributed_llm::engines::sglang;
            let Some(model_path) = model_path else {
                anyhow::bail!("out=sglang requires flag --model-path=<full-path-to-model-dir>");
            };
            if !model_path.is_dir() {
                anyhow::bail!("`--model-path should point at a HuggingFace repo checkout");
            }
            // Safety: Earlier we build maybe_card from model_path, which we checked right above
            let card = maybe_card.clone().unwrap();
            let Some(sock_prefix) = zmq_socket_prefix else {
                anyhow::bail!("sglang requires zmq_socket_prefix");
            };
            let node_conf = sglang::MultiNodeConfig {
                num_nodes: flags.num_nodes,
                node_rank: flags.node_rank,
                dist_init_addr: flags.dist_init_addr,
            };
            if node_conf.num_nodes > 1 {
                if let Ok(Some(if_name)) = net::get_primary_interface().await {
                    tracing::info!("If you see 'gloo' errors from sglang try setting these environment variables:");
                    tracing::info!("export GLOO_SOCKET_IFNAME={if_name}");
                    tracing::info!("export NCCL_SOCKET_IFNAME={if_name}");
                }
            }

            let (engine, sglang_process) = sglang::make_engine(
                cancel_token.clone(),
                &model_path,
                &sock_prefix,
                node_conf,
                flags.tensor_parallel_size,
                flags.base_gpu_id,
            )
            .await?;
            extra = Some(sglang_process);
            EngineConfig::StaticCore {
                service_name: card.service_name.clone(),
                engine,
                card: Box::new(card),
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
        Input::None => {
            // Multi-node setup. The engine sub-process has been started and is talking
            // to it's node_rank 0 controller. We do nothing.
            // TODO: Acquire an etcd lease, we are running
            cancel_token.cancelled().await;
        }
    }

    #[cfg(feature = "sglang")]
    // Allow engines to ask main thread to wait on an extra future.
    // sglang uses this to shut down sub-process
    if let Some(extra) = extra {
        extra.await?;
    }

    Ok(())
}
