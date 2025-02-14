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

use clap::{Parser, Subcommand};
use tracing as log;

use triton_distributed::{
    distributed::DistributedConfig, logging, protocols::Endpoint, raise, DistributedRuntime,
    Result, Runtime, Worker,
};
use triton_llm::http::service::discovery::ModelEntry;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Namespace to operate in
    #[arg(short = 'n', long)]
    namespace: Option<String>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// HTTP service related commands
    Http {
        #[command(subcommand)]
        command: HttpCommands,
    },
}

#[derive(Subcommand)]
enum HttpCommands {
    /// Add a chat model
    Add {
        /// Specifies we're adding a chat model
        #[arg(value_name = "chat-model")]
        chat_model: String,

        /// Model name (e.g. foo/v1)
        model_name: String,

        /// Endpoint name (format: component.endpoint or namespace.component.endpoint)
        endpoint_name: String,
    },

    /// List chat models
    List {
        /// Specifies we're listing chat models
        #[arg(value_name = "chat-model", value_parser = parse_chat_model)]
        chat_model: String,
    },

    /// Remove a chat model
    Remove {
        /// Specifies we're removing a chat model
        #[arg(value_name = "chat-model")]
        chat_model: String,

        /// Name of the model to remove
        name: String,
    },
}

fn parse_chat_model(s: &str) -> Result<String> {
    match s {
        "chat-model" | "chat-models" => Ok(s.to_string()),
        _ => raise!("Expected 'chat-model' or 'chat-models'"),
    }
}

fn main() -> Result<()> {
    logging::init();
    let cli = Cli::parse();

    // Default namespace to "public" if not specified
    let namespace = cli.namespace.unwrap_or_else(|| "public".to_string());

    let worker = Worker::from_settings()?;
    worker.execute(|runtime| async move { handle_command(runtime, namespace, cli.command).await })
}

async fn handle_command(runtime: Runtime, namespace: String, command: Commands) -> Result<()> {
    let settings = DistributedConfig::for_cli();
    let distributed = DistributedRuntime::new(runtime, settings).await?;

    match command {
        Commands::Http { command } => {
            match command {
                HttpCommands::Add {
                    chat_model: _,
                    model_name,
                    endpoint_name,
                } => {
                    log::debug!(
                        "Adding model {} with endpoint {}",
                        model_name,
                        endpoint_name
                    );

                    // parse endpoint
                    // split by '.' must have 2, can have 3 parts, any more or less is an error
                    let parts: Vec<&str> = endpoint_name.split('.').collect();
                    if parts.len() < 2 || parts.len() > 3 {
                        raise!("Invalid endpoint name: {}", endpoint_name);
                    }

                    // if 3 parts, then it's namespace.component.endpoint
                    // if 2 parts, then it's model_name.component.endpoint

                    // create model entry
                    let endpoint = Endpoint {
                        namespace: if parts.len() == 3 {
                            parts[0].to_string()
                        } else {
                            namespace.clone()
                        },
                        component: parts[parts.len() - 2].to_string(),
                        name: parts[parts.len() - 1].to_string(),
                    };

                    let model = ModelEntry {
                        name: model_name.clone(),
                        endpoint,
                    };

                    // add model to etcd
                    let component = distributed.namespace(&namespace)?.component("http")?;
                    let path = format!("{}/models/chat/{}", component.etcd_path(), model_name);
                    let etcd_client = distributed.etcd_client();

                    etcd_client
                        .kv_create(path, serde_json::to_vec_pretty(&model)?.into(), None)
                        .await?;

                    println!("Model {} added to namespace {}", model_name, namespace);
                }
                HttpCommands::List { chat_model: _ } => {
                    let component = distributed.namespace(&namespace)?.component("http")?;
                    // todo - make this part of the http discovery service object
                    let prefix = format!("{}/models/chat/", component.etcd_path());

                    // get the kvs from etcd
                    let etcd_client = distributed.etcd_client();
                    let kvs = etcd_client.kv_get_prefix(&prefix).await?;

                    use tabled::Tabled;

                    #[derive(Tabled)]
                    struct ModelRow {
                        #[tabled(rename = "MODEL NAME")]
                        name: String,
                        #[tabled(rename = "NAMESPACE")]
                        namespace: String,
                        #[tabled(rename = "COMPONENT")]
                        component: String,
                        #[tabled(rename = "ENDPOINT")]
                        endpoint: String,
                    }

                    // parse the keys
                    let mut models = Vec::new();
                    for kv in kvs {
                        match (
                            kv.key_str(),
                            serde_json::from_slice::<ModelEntry>(&kv.value()),
                        ) {
                            (Ok(key), Ok(model)) => {
                                models.push(ModelRow {
                                    name: key.trim_start_matches(&prefix).to_string(),
                                    namespace: model.endpoint.namespace,
                                    component: model.endpoint.component,
                                    endpoint: model.endpoint.name,
                                });
                            }
                            (Err(e), _) => {
                                log::debug!("Error parsing key: {}", e);
                            }
                            (_, Err(e)) => {
                                log::debug!("Error parsing value: {}", e);
                            }
                        }
                    }

                    if models.is_empty() {
                        println!("No chat models found in namespace {}", namespace);
                    } else {
                        let table = tabled::Table::new(models);
                        println!("Listing chat models in namespace {}", namespace);
                        println!("{}", table);
                    }
                }
                HttpCommands::Remove {
                    chat_model: _,
                    name,
                } => {
                    // TODO: Implement remove logic
                    log::debug!("Removing model {}", name);

                    let component = distributed.namespace(&namespace)?.component("http")?;
                    // todo - make this part of the http discovery service object
                    let prefix = format!("{}/models/chat/{name}", component.etcd_path());

                    log::debug!("deleting key: {}", prefix);

                    // get the kvs from etcd
                    let mut kv_client = distributed.etcd_client().etcd_client().kv_client();
                    match kv_client.delete(prefix.as_bytes(), None).await {
                        Ok(_response) => {
                            println!("Model {} removed from namespace {}", name, namespace);
                        }
                        Err(e) => {
                            log::error!("Error removing model {}: {}", name, e);
                        }
                    }
                }
            }
        }
    }

    Ok(())
}
