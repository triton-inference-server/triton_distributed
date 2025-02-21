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

use std::{str::FromStr, sync::Arc};

use serde::{Deserialize, Serialize};
use tokio::sync::mpsc::Receiver;
use tracing as log;

use triton_distributed::{
    error,
    protocols::{self, annotated::Annotated, EndpointAddress},
    raise,
    transports::etcd::{KeyValue, WatchEvent},
    DistributedRuntime, Result,
};

use super::ModelManager;
use crate::protocols::openai::chat_completions::{
    ChatCompletionRequest, ChatCompletionResponseDelta,
};

#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub enum ModelState {
    Ready,
    Unavailable,
}

/// [ModelEntry] is a struct that contains the information for the HTTP service to discover models
/// from the etcd cluster.
#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct ModelEntry {
    /// Public name of the model
    /// This will be used to identify the model in the HTTP service and the value used in an
    /// an [OAI ChatRequest][crate::protocols::openai::chat_completions::ChatCompletionRequest].
    pub name: String,

    /// Component of the endpoint.
    pub endpoint: protocols::Endpoint,

    // TODO - deprecate default
    // default is required now to support older versions of the model entry otherwise deserialization
    // will fail. by release, remove the option on ModelState and make it required
    #[serde(default = "default_model_state")]
    pub state: Option<ModelState>,
}

fn default_model_state() -> Option<ModelState> {
    Some(ModelState::Ready)
}

pub struct ModelWatchState {
    pub prefix: String,
    pub manager: ModelManager,
    pub drt: DistributedRuntime,
}

pub async fn model_watcher(state: Arc<ModelWatchState>, events_rx: Receiver<WatchEvent>) {
    log::debug!("model watcher started");

    let mut events_rx = events_rx;

    while let Some(event) = events_rx.recv().await {
        match event {
            WatchEvent::Put(kv) => match handle_put(&kv, state.clone()).await {
                Ok(_) => {}
                Err(e) => {
                    log::error!("error adding chat model: {}", e);
                }
            },
            WatchEvent::Delete(kv) => match handle_delete(&kv, state.clone()).await {
                Ok(model_name) => {
                    log::info!("removed chat model: {}", model_name);
                }
                Err(e) => {
                    log::error!("error removing chat model: {}", e);
                }
            },
        }
    }

    log::debug!("model watcher stopped");
}

async fn handle_delete(kv: &KeyValue, state: Arc<ModelWatchState>) -> Result<String> {
    log::debug!("removing model");

    let key = kv.key_str()?;
    log::debug!("key: {}", key);

    let model_name = key.trim_start_matches(&state.prefix);
    state.manager.remove_chat_completions_model(model_name)?;
    Ok(model_name.to_string())
}

// Handles a PUT event from etcd, this usually means adding a new model to the list of served
// models.
//
// If this method errors, for the near term, we will delete the offending key.
async fn handle_put(kv: &KeyValue, state: Arc<ModelWatchState>) -> Result<()> {
    log::debug!("adding model");

    let key = kv.key_str()?;
    log::debug!("key: {}", key);

    let path = key.trim_start_matches(&state.prefix);

    let (model_name, action) = path
        .split_once('/')
        .ok_or(error!("invalid key entry: {}", key))?;

    log::debug!("model_name: {}", model_name);

    // Skip state keys for now
    if action.ends_with("state") {
        log::debug!("skipping state update for now");
        return Ok(());
    }

    // Only process .address keys
    if !action.ends_with(".address") {
        raise!("invalid key entry: {}", key);
    }

    let (request_format, _) = action
        .split_once(".")
        .ok_or(error!("invalid key entry: {}", key))?;

    // Parse the endpoint address from the value
    let endpoint_addr = EndpointAddress::from_str(kv.value_str()?)?;
    let endpoint_addr: protocols::Endpoint = endpoint_addr.into();
    let (namespace, component, endpoint) = endpoint_addr.dissolve();

    let client = state
        .drt
        .namespace(&namespace)?
        .component(component)?
        .endpoint(endpoint)
        .client::<ChatCompletionRequest, Annotated<ChatCompletionResponseDelta>>()
        .await?;

    let client = Arc::new(client);

    state
        .manager
        .add_chat_completions_model(model_name, client)?;

    Ok(())
}
