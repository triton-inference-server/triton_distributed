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

use triton_distributed::runtime::CancellationToken;
use triton_llm::types::openai::chat_completions::OpenAIChatCompletionsStreamingEngine;

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
    /// Later that will come from the HF repo name, and still later from etcd during discovery
    #[arg(long)]
    pub model_name: String,
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
    // Create the engine matching `out`
    let engine_config = match out_opt {
        Output::EchoFull => EngineConfig::StaticFull {
            service_name: flags.model_name,
            engine: output::echo_full::make_engine_full(),
        },
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
