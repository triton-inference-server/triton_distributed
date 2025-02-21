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

use std::env;

use clap::Parser;

use tio::{Input, Output};
use triton_distributed::logging;

const HELP: &str = r#"
tio is a single binary that wires together the various inputs (http, text, network) and workers (network, engine), that runs the services. It is the simplest way to use triton-distributed locally.

Example:
- cargo build --release --features mistralrs,cuda
- cd target/release
- ./tio hf_checkouts/Llama-3.2-3B-Instruct/
- OR: ./tio Llama-3.2-1B-Instruct-Q4_K_M.gguf

"#;

const DEFAULT_IN: Input = Input::Text;

#[cfg(feature = "mistralrs")]
const DEFAULT_OUT: Output = Output::MistralRs;

#[cfg(not(feature = "mistralrs"))]
const DEFAULT_OUT: Output = Output::EchoFull;

const USAGE: &str = "USAGE: tio in=[http|text] out=[mistralrs|echo_full] [--http-port 8080] [--model-path <path>] [--model-name <served-model-name>]";

fn main() -> anyhow::Result<()> {
    logging::init();

    // max_worker_threads and max_blocking_threads from env vars or config file.
    let rt_config = triton_distributed::RuntimeConfig::from_settings()?;

    // One per process. Wraps a Runtime with holds two tokio runtimes.
    let worker = triton_distributed::Worker::from_config(rt_config)?;

    worker.execute(tio_wrapper)
}

async fn tio_wrapper(runtime: triton_distributed::Runtime) -> anyhow::Result<()> {
    let mut in_opt = None;
    let mut out_opt = None;
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() || args[0] == "-h" || args[0] == "--help" {
        println!("{USAGE}");
        println!("{HELP}");
        return Ok(());
    }
    for arg in env::args().skip(1).take(2) {
        let Some((in_out, val)) = arg.split_once('=') else {
            // Probably we're defaulting in and/or out, and this is a flag
            continue;
        };
        match in_out {
            "in" => {
                in_opt = Some(val.try_into()?);
            }
            "out" => {
                out_opt = Some(val.try_into()?);
            }
            _ => {
                anyhow::bail!("Invalid argument, must start with 'in' or 'out. {USAGE}");
            }
        }
    }
    let mut non_flag_params = 1; // binary name
    let in_opt = match in_opt {
        Some(x) => {
            non_flag_params += 1;
            x
        }
        None => DEFAULT_IN,
    };
    let out_opt = match out_opt {
        Some(x) => {
            non_flag_params += 1;
            x
        }
        None => DEFAULT_OUT,
    };

    // Clap skips the first argument expecting it to be the binary name, so add it back
    // Note `--model-path` has index=1 (in lib.rs) so that doesn't need a flag.
    let flags = tio::Flags::try_parse_from(
        ["tio".to_string()]
            .into_iter()
            .chain(env::args().skip(non_flag_params)),
    )?;

    // etcd and nats addresses, from env vars ETCD_ENDPOINTS and NATS_SERVER with localhost
    // defaults
    //let dt_config = triton_distributed::distributed::DistributedConfig::from_settings();
    // Wraps the Runtime (which wraps two tokio runtimes) and adds etcd and nats clients
    //let d_runtime = triton_distributed::DistributedRuntime::new(runtime, dt_config).await?;

    tio::run(in_opt, out_opt, flags, runtime.primary_token()).await
}
