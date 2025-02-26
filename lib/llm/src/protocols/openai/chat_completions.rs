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
// use derive_builder::Builder;
// use serde::de::{self, SeqAccess, Visitor};
// use serde::ser::SerializeMap;
use serde::{Deserialize, Serialize};
// use serde::{Deserializer, Serializer};
// use std::collections::HashMap;
// use std::fmt;
// use std::fmt::Display;
use triton_distributed_runtime::protocols::annotated::AnnotationsProvider;
use validator::Validate;

mod aggregator;
mod delta;

pub use super::{CompletionTokensDetails, CompletionUsage, PromptTokensDetails};
// pub use aggregator::DeltaAggregator;
// pub use delta::DeltaGenerator;

// use super::{common::ChatCompletionLogprobs, ContentProvider};

/// Request object which is used to generate chat completions.
#[derive(Serialize, Deserialize, Validate, Debug, Clone)]
pub struct ChatCompletionRequest {
    #[serde(flatten)]
    pub inner: async_openai::types::CreateChatCompletionRequest,
    pub nvext: Option<NvExt>,
}

/// Request object which is used to generate chat completions.
#[derive(Serialize, Deserialize, Validate, Debug, Clone)]
pub struct ChatCompletionResponse {
    #[serde(flatten)]
    pub inner: async_openai::types::CreateChatCompletionResponse,
    pub nvext: Option<NvExt>,
}

/// Request object which is used to generate chat completions.
#[derive(Serialize, Deserialize, Validate, Debug, Clone)]
pub struct ChatCompletionContent {
    #[serde(flatten)]
    pub inner: async_openai::types::ChatCompletionStreamResponseDelta,
    pub nvext: Option<NvExt>,
}

/// Request object which is used to generate chat completions.
#[derive(Serialize, Deserialize, Validate, Debug, Clone)]
pub struct ChatCompletionResponseDelta {
    #[serde(flatten)]
    pub inner: async_openai::types::CreateChatCompletionStreamResponse,
    pub nvext: Option<NvExt>,
}

impl NvExtProvider for ChatCompletionRequest {
    fn nvext(&self) -> Option<&NvExt> {
        self.nvext.as_ref()
    }

    fn raw_prompt(&self) -> Option<String> {
        None
    }
}

impl AnnotationsProvider for ChatCompletionRequest {
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

impl OpenAISamplingOptionsProvider for ChatCompletionRequest {
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

impl OpenAIStopConditionsProvider for ChatCompletionRequest {
    fn get_max_tokens(&self) -> Option<u32> {
        // TODO THIS IS WRONG i32 -> u32
        // self.chat_completion_request.max_tokens
        None
    }

    fn get_min_tokens(&self) -> Option<u32> {
        // TODO THIS IS WRONG min_tokens does not exist
        None
        // self.chat_completion_request.min_tokens
    }

    fn get_stop(&self) -> Option<Vec<String>> {
        // TODO THIS IS WRONG Vec<String> -> Stop
        // self.chat_completion_request.stop.clone()
        None
    }

    fn nvext(&self) -> Option<&NvExt> {
        self.nvext.as_ref()
    }
}
