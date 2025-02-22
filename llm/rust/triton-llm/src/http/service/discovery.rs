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

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio::sync::mpsc::Receiver;

use triton_distributed::{
    protocols::{self, annotated::Annotated},
    transports::etcd::{KeyValue, WatchEvent},
    DistributedRuntime, Result, raise,
};

use super::ModelManager;
use crate::protocols::openai::chat_completions::{
    ChatCompletionRequest, ChatCompletionResponseDelta,
};
use crate::protocols::openai::completions::{
    CompletionRequest, CompletionResponse
};
use crate::model_type::ModelType;
use tracing as log;

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

    /// Specifies whether the model is a chat or completion model.s
    pub model_type: ModelType,
}

pub struct ModelWatchState {
    pub prefix_to_name: String,
    pub prefix_to_type: String,
    pub manager: ModelManager,
    pub drt: DistributedRuntime,
}

pub async fn model_watcher(state: Arc<ModelWatchState>, events_rx: Receiver<WatchEvent>) {
    tracing::debug!("model watcher started");

    let mut events_rx = events_rx;

    while let Some(event) = events_rx.recv().await {
        match event {
            WatchEvent::Put(kv) => match handle_put(&kv, state.clone()).await {
                Ok((model_name, model_type)) => {
                    log::info!("added {} model: {}",model_type, model_name);
                }
                Err(e) => {
                    log::error!("error adding model: {}", e);
                }
            },
            WatchEvent::Delete(kv) => match handle_delete(&kv, state.clone()).await {
                Ok((model_name, model_type)) => {
                    log::info!("removed {} model: {}", model_type, model_name);
                }
                Err(e) => {
                    log::error!("error removing model: {}", e);
                }
            },
        }
    }

    tracing::debug!("model watcher stopped");
}

async fn handle_delete(kv: &KeyValue, state: Arc<ModelWatchState>) -> Result<(String, String)> {
    log::debug!("removing model");

    let key = kv.key_str()?;
    tracing::debug!("key: {}", key);

    let model_name = key.trim_start_matches(&state.prefix_to_name);
    let model_type = key.trim_start_matches(&state.prefix_to_type)
        .trim_end_matches(&format!("/{}", model_name));

    match model_type {
        "chat" => state.manager.remove_chat_completions_model(model_name)?,
        "completion" => state.manager.remove_completions_model(model_name)?,
        unknown => log::warn!("Unknown model type: {}. No Action Taken.", unknown),
    };

    Ok((model_name.to_string(), model_type.to_string()))
}

// Handles a PUT event from etcd, this usually means adding a new model to the list of served
// models.
//
// If this method errors, for the near term, we will delete the offending key.
async fn handle_put(kv: &KeyValue, state: Arc<ModelWatchState>) -> Result<(String, String)> {
    log::debug!("adding model");

    let key = kv.key_str()?;
    tracing::debug!("key: {}", key);

    let model_name = key.trim_start_matches(&state.prefix_to_name);
    let model_type = key.trim_start_matches(&state.prefix_to_type)
        .trim_end_matches(&format!("/{}", model_name));
    let model_entry = serde_json::from_slice::<ModelEntry>(kv.value())?;

    if model_entry.name != model_name {
        raise!(
            "model name mismatch: {} != {}",
            model_entry.name,
            model_name
        );
    }

    match model_type {
        "chat" => {
            let client = state
                .drt
                .namespace(model_entry.endpoint.namespace)?
                .component(model_entry.endpoint.component)?
                .endpoint(model_entry.endpoint.name)
                .client::<ChatCompletionRequest, Annotated<ChatCompletionResponseDelta>>()
                .await?;
            state.manager.add_chat_completions_model(model_name, Arc::new(client))?;
        }
        "completion" => {
            let client = state
                .drt
                .namespace(model_entry.endpoint.namespace)?
                .component(model_entry.endpoint.component)?
                .endpoint(model_entry.endpoint.name)
                .client::<CompletionRequest, Annotated<CompletionResponse>>()
                .await?;
            state.manager.add_completions_model(model_name, Arc::new(client))?;
        }
        unknown => log::warn!("Unknown model type: {}. No Action Taken.", unknown),
    }

    Ok((model_name.to_string(), model_type.to_string()))
}
