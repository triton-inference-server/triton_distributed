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

use super::nvext::NvExt;
use super::nvext::NvExtProvider;
use super::OpenAISamplingOptionsProvider;
use super::OpenAIStopConditionsProvider;
use serde::{Deserialize, Serialize};
use triton_distributed_runtime::protocols::annotated::AnnotationsProvider;
use validator::Validate;

mod aggregator;
mod delta;

pub use aggregator::DeltaAggregator;
pub use delta::DeltaGenerator;

#[derive(Serialize, Deserialize, Validate, Debug, Clone)]
pub struct NvCreateChatCompletionRequest {
    #[serde(flatten)]
    pub inner: async_openai::types::CreateChatCompletionRequest,
    pub nvext: Option<NvExt>,
}

#[derive(Serialize, Deserialize, Validate, Debug, Clone)]
pub struct NvCreateChatCompletionResponse {
    #[serde(flatten)]
    pub inner: async_openai::types::CreateChatCompletionResponse,
}

#[derive(Serialize, Deserialize, Validate, Debug, Clone)]
pub struct ChatCompletionContent {
    #[serde(flatten)]
    pub inner: async_openai::types::ChatCompletionStreamResponseDelta,
}

#[derive(Serialize, Deserialize, Validate, Debug, Clone)]
pub struct NvCreateChatCompletionStreamResponse {
    #[serde(flatten)]
    pub inner: async_openai::types::CreateChatCompletionStreamResponse,
}

impl NvExtProvider for NvCreateChatCompletionRequest {
    fn nvext(&self) -> Option<&NvExt> {
        self.nvext.as_ref()
    }

    fn raw_prompt(&self) -> Option<String> {
        None
    }
}

impl AnnotationsProvider for NvCreateChatCompletionRequest {
    fn annotations(&self) -> Option<Vec<String>> {
        self.nvext
            .as_ref()
            .and_then(|nvext| nvext.annotations.clone())
    }

    fn has_annotation(&self, annotation: &str) -> bool {
        self.nvext
            .as_ref()
            .and_then(|nvext| nvext.annotations.as_ref())
            .map(|annotations| annotations.contains(&annotation.to_string()))
            .unwrap_or(false)
    }
}

impl OpenAISamplingOptionsProvider for NvCreateChatCompletionRequest {
    fn get_temperature(&self) -> Option<f32> {
        self.inner.temperature
    }

    fn get_top_p(&self) -> Option<f32> {
        self.inner.top_p
    }

    fn get_frequency_penalty(&self) -> Option<f32> {
        self.inner.frequency_penalty
    }

    fn get_presence_penalty(&self) -> Option<f32> {
        self.inner.presence_penalty
    }

    fn nvext(&self) -> Option<&NvExt> {
        self.nvext.as_ref()
    }
}

#[allow(deprecated)]
impl OpenAIStopConditionsProvider for NvCreateChatCompletionRequest {
    fn get_max_tokens(&self) -> Option<u32> {
        // ALLOW: max_tokens is deprecated in favor of max_completion_tokens
        self.inner.max_tokens
    }

    fn get_min_tokens(&self) -> Option<u32> {
        // TODO THIS IS WRONG min_tokens does not exist
        None
    }

    fn get_stop(&self) -> Option<Vec<String>> {
        // TODO THIS IS WRONG should instead do
        // Vec<String> -> async_openai::types::Stop
        // self.inner.stop.clone()
        None
    }

    fn nvext(&self) -> Option<&NvExt> {
        self.nvext.as_ref()
    }
}
