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

use super::{
    ChatCompletionChoiceDelta, ChatCompletionContent, ChatCompletionRequest,
    ChatCompletionResponseDelta, FinishReason, MessageRole, ServiceTier,
};
use crate::protocols::common;
use crate::protocols::openai::CompletionUsage;

impl ChatCompletionRequest {
    // put this method on the request
    // inspect the request to extract options
    pub fn response_generator(&self) -> DeltaGenerator {
        let options = DeltaGeneratorOptions {
            enable_usage: true,
            enable_logprobs: self.chat_completion_request.logprobs.unwrap_or(false),
        };

        DeltaGenerator::new(self.chat_completion_request.model.clone(), options)
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
    created: u64,
    model: String,
    system_fingerprint: Option<String>,
    service_tier: Option<ServiceTier>,
    usage: CompletionUsage,

    // counter on how many messages we have issued
    msg_counter: u64,

    options: DeltaGeneratorOptions,
}

impl DeltaGenerator {
    pub fn new(model: String, options: DeltaGeneratorOptions) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Self {
            id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
            object: "chat.completion.chunk".to_string(),
            created: now,
            model,
            system_fingerprint: None,
            service_tier: None,
            usage: CompletionUsage::default(),
            msg_counter: 0,
            options,
        }
    }

    pub fn update_isl(&mut self, isl: i32) {
        self.usage.prompt_tokens = isl;
    }

    pub fn create_choice(
        &self,
        index: u64,
        text: Option<String>,
        finish_reason: Option<super::FinishReason>,
        logprobs: Option<super::ChatCompletionLogprobs>,
    ) -> ChatCompletionResponseDelta {
        // todo - update for tool calling
        let delta = ChatCompletionContent {
            content: text,
            role: if self.msg_counter == 0 {
                Some(MessageRole::assistant)
            } else {
                None
            },
            tool_calls: None,
        };

        ChatCompletionResponseDelta {
            id: self.id.clone(),
            object: self.object.clone(),
            created: self.created,
            model: self.model.clone(),
            system_fingerprint: self.system_fingerprint.clone(),
            choices: vec![ChatCompletionChoiceDelta {
                index,
                delta,
                finish_reason,
                logprobs,
            }],
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
            self.usage.completion_tokens += delta.token_ids.len() as i32;
        }

        // todo logprobs
        let logprobs = None;

        let finish_reason = match delta.finish_reason {
            Some(common::FinishReason::EoS) => Some(FinishReason::stop),
            Some(common::FinishReason::Stop) => Some(FinishReason::stop),
            Some(common::FinishReason::Length) => Some(FinishReason::length),
            Some(common::FinishReason::Cancelled) => Some(FinishReason::cancelled),
            Some(common::FinishReason::Error(err_msg)) => {
                return Err(anyhow::anyhow!(err_msg));
            }
            None => None,
        };

        // create choice
        let index = 0;
        Ok(self.create_choice(index, delta.text, finish_reason, logprobs))
    }
}
