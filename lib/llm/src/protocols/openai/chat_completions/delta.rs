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

use super::{ChatCompletionResponseDelta, NvCreateChatCompletionRequest};
use crate::protocols::common;

impl NvCreateChatCompletionRequest {
    // put this method on the request
    // inspect the request to extract options
    pub fn response_generator(&self) -> DeltaGenerator {
        let options = DeltaGeneratorOptions {
            enable_usage: true,
            enable_logprobs: self.inner.logprobs.unwrap_or(false),
        };

        DeltaGenerator::new(self.inner.model.clone(), options)
    }
}

#[derive(Debug, Clone, Default)]
pub struct DeltaGeneratorOptions {
    pub enable_usage: bool,
    pub enable_logprobs: bool,
}

#[derive(Debug, Clone)]
pub struct DeltaGenerator {
    id: String,
    object: String,
    created: u32,
    model: String,
    system_fingerprint: Option<String>,
    service_tier: Option<async_openai::types::ServiceTierResponse>,
    usage: async_openai::types::CompletionUsage,

    // counter on how many messages we have issued
    msg_counter: u64,

    options: DeltaGeneratorOptions,
}

impl DeltaGenerator {
    pub fn new(model: String, options: DeltaGeneratorOptions) -> Self {
        // SAFETY: This is a fun one to write. We are casting from u64 to u32
        // which typically is unsafe due to loss of precision after it
        // exceeds u32::MAX. Fortunately, this won't be an issue until
        // 2106. So whoever is still maintaining this then, enjoy!
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as u32;

        let usage = async_openai::types::CompletionUsage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
            prompt_tokens_details: None,
            completion_tokens_details: None,
        };

        Self {
            id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
            object: "chat.completion.chunk".to_string(),
            created: now,
            model,
            system_fingerprint: None,
            service_tier: None,
            usage,
            msg_counter: 0,
            options,
        }
    }

    pub fn update_isl(&mut self, isl: u32) {
        self.usage.prompt_tokens = isl;
    }

    #[allow(deprecated)]
    pub fn create_choice(
        &self,
        index: u32,
        text: Option<String>,
        finish_reason: Option<async_openai::types::FinishReason>,
        logprobs: Option<async_openai::types::ChatChoiceLogprobs>,
    ) -> async_openai::types::CreateChatCompletionStreamResponse {
        // TODO: Update for tool calling
        // ALLOW: function_call is deprecated
        let delta = async_openai::types::ChatCompletionStreamResponseDelta {
            role: if self.msg_counter == 0 {
                Some(async_openai::types::Role::Assistant)
            } else {
                None
            },
            content: text,
            tool_calls: None,
            function_call: None,
            refusal: None,
        };

        let choice = async_openai::types::ChatChoiceStream {
            index,
            delta,
            finish_reason,
            logprobs,
        };

        let choices = vec![choice];

        async_openai::types::CreateChatCompletionStreamResponse {
            id: self.id.clone(),
            object: self.object.clone(),
            created: self.created,
            model: self.model.clone(),
            system_fingerprint: self.system_fingerprint.clone(),
            choices,
            usage: if self.options.enable_usage {
                Some(self.usage.clone())
            } else {
                None
            },
            service_tier: self.service_tier.clone(),
        }
    }
}

impl crate::protocols::openai::DeltaGeneratorExt<ChatCompletionResponseDelta> for DeltaGenerator {
    fn choice_from_postprocessor(
        &mut self,
        delta: crate::protocols::common::llm_backend::BackendOutput,
    ) -> anyhow::Result<ChatCompletionResponseDelta> {
        // aggregate usage
        if self.options.enable_usage {
            self.usage.completion_tokens += delta.token_ids.len() as u32;
        }

        // todo logprobs
        let logprobs = None;

        let finish_reason = match delta.finish_reason {
            Some(common::FinishReason::EoS) => Some(async_openai::types::FinishReason::Stop),
            Some(common::FinishReason::Stop) => Some(async_openai::types::FinishReason::Stop),
            Some(common::FinishReason::Length) => Some(async_openai::types::FinishReason::Length),
            Some(common::FinishReason::Cancelled) => Some(async_openai::types::FinishReason::Stop),
            Some(common::FinishReason::Error(err_msg)) => {
                return Err(anyhow::anyhow!(err_msg));
            }
            None => None,
        };

        // create choice
        let index = 0;
        let stream_response = self.create_choice(index, delta.text, finish_reason, logprobs);

        Ok(ChatCompletionResponseDelta {
            inner: stream_response,
        })
    }
}
