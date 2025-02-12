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

use std::{
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use async_stream::stream;
use async_trait::async_trait;

use triton_distributed::engine::{AsyncEngine, AsyncEngineContextProvider, ResponseStream};
use triton_distributed::pipeline::{Error, ManyOut, SingleIn};
use triton_distributed::protocols::annotated::Annotated;
use triton_llm::protocols::openai::chat_completions::FinishReason;
use triton_llm::protocols::openai::chat_completions::{
    ChatCompletionChoiceDelta, ChatCompletionContent, ChatCompletionRequest,
    ChatCompletionResponseDelta, Content, MessageRole,
};
use triton_llm::types::openai::chat_completions::OpenAIChatCompletionsStreamingEngine;

/// How long to sleep between echoed tokens.
/// 50ms gives us 20 tok/s.
const TOKEN_ECHO_DELAY: Duration = Duration::from_millis(50);

/// Engine that accepts un-preprocessed requests and echos the prompt back as the response
/// Useful for testing ingress such as service-http.
struct EchoEngineFull {}
pub fn make_engine_full() -> OpenAIChatCompletionsStreamingEngine {
    Arc::new(EchoEngineFull {})
}

#[async_trait]
impl
    AsyncEngine<
        SingleIn<ChatCompletionRequest>,
        ManyOut<Annotated<ChatCompletionResponseDelta>>,
        Error,
    > for EchoEngineFull
{
    async fn generate(
        &self,
        incoming_request: SingleIn<ChatCompletionRequest>,
    ) -> Result<ManyOut<Annotated<ChatCompletionResponseDelta>>, Error> {
        let (request, context) = incoming_request.transfer(());
        let ctx = context.context();
        let req = request.messages.into_iter().last().unwrap();
        let prompt = match req.content {
            Content::Text(prompt) => prompt,
            _ => {
                anyhow::bail!("Invalid request content field, expected Content::Text");
            }
        };
        let output = stream! {
            let mut id = 1;
            for c in prompt.chars() {
                // we are returning characters not tokens, so speed up some
                tokio::time::sleep(TOKEN_ECHO_DELAY/2).await;
                yield delta_full(id, c.to_string());
                id += 1;
            }
            yield stop(id);
        };
        Ok(ResponseStream::new(Box::pin(output), ctx))
    }
}

fn delta_full(id: usize, token: String) -> Annotated<ChatCompletionResponseDelta> {
    create_delta(id, token, None)
}

fn stop(id: usize) -> Annotated<ChatCompletionResponseDelta> {
    create_delta(id, "".to_string(), Some(FinishReason::stop))
}

fn create_delta(
    id: usize,
    token: String,
    finish_reason: Option<FinishReason>,
) -> Annotated<ChatCompletionResponseDelta> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let delta = ChatCompletionResponseDelta {
        id: id.to_string(),
        choices: vec![ChatCompletionChoiceDelta {
            index: 0,
            delta: ChatCompletionContent {
                role: Some(MessageRole::assistant),
                content: Some(token),
                tool_calls: None,
            },
            logprobs: None,
            finish_reason,
        }],
        model: "echo".to_string(),
        created: now,
        object: "text_completion".to_string(),
        usage: None,
        system_fingerprint: None,
        service_tier: None,
    };
    Annotated {
        id: None,
        data: Some(delta),
        event: None,
        comment: None,
    }
}
