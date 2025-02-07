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
    ChatCompletionChoice, ChatCompletionContent, ChatCompletionResponse,
    ChatCompletionResponseDelta, CompletionUsage, FinishReason, MessageRole, ServiceTier,
};
use crate::protocols::{
    codec::{Message, SseCodecError},
    common::ChatCompletionLogprobs,
    convert_sse_stream, Annotated,
};

use futures::{Stream, StreamExt};
use std::{collections::HashMap, pin::Pin};

type DataStream<T> = Pin<Box<dyn Stream<Item = T> + Send + Sync>>;

/// Aggregates a stream of [`ChatCompletionResponseDelta`]s into a single [`ChatCompletionResponse`].
pub struct DeltaAggregator {
    id: String,
    model: String,
    created: u64,
    usage: Option<CompletionUsage>,
    system_fingerprint: Option<String>,
    choices: HashMap<u64, DeltaChoice>,
    error: Option<String>,
    service_tier: Option<ServiceTier>,
}

// Holds the accumulated state of a choice
struct DeltaChoice {
    index: u64,
    text: String,
    role: Option<MessageRole>,
    finish_reason: Option<FinishReason>,
    logprobs: Option<ChatCompletionLogprobs>,
}

impl Default for DeltaAggregator {
    fn default() -> Self {
        Self::new()
    }
}

impl DeltaAggregator {
    /// Creates a new [`DeltaAggregator`].
    pub fn new() -> Self {
        Self {
            id: "".to_string(),
            model: "".to_string(),
            created: 0,
            usage: None,
            system_fingerprint: None,
            choices: HashMap::new(),
            error: None,
            service_tier: None,
        }
    }

    /// Aggregates a stream of [`ChatCompletionResponseDelta`]s into a single [`ChatCompletionResponse`].
    pub async fn apply(
        stream: DataStream<Annotated<ChatCompletionResponseDelta>>,
    ) -> Result<ChatCompletionResponse, String> {
        let aggregator = stream
            .fold(DeltaAggregator::new(), |mut aggregator, delta| async move {
                // these are cheap to move so we do it every time since we are consuming the delta

                let delta = match delta.ok() {
                    Ok(delta) => delta,
                    Err(error) => {
                        aggregator.error = Some(error);
                        return aggregator;
                    }
                };

                if aggregator.error.is_none() && delta.data.is_some() {
                    // note: we could extract annotations here and add them to the aggregator
                    // to be return as part of the NIM Response Extension
                    // TODO(#14) - Aggregate Annotation

                    let delta = delta.data.unwrap();
                    aggregator.id = delta.id;
                    aggregator.model = delta.model;
                    aggregator.created = delta.created;
                    aggregator.service_tier = delta.service_tier;
                    if let Some(usage) = delta.usage {
                        aggregator.usage = Some(usage);
                    }
                    if let Some(system_fingerprint) = delta.system_fingerprint {
                        aggregator.system_fingerprint = Some(system_fingerprint);
                    }

                    // handle the choices
                    for choice in delta.choices {
                        let state_choice =
                            aggregator
                                .choices
                                .entry(choice.index)
                                .or_insert(DeltaChoice {
                                    index: choice.index,
                                    text: "".to_string(),
                                    role: choice.delta.role,
                                    finish_reason: None,
                                    logprobs: choice.logprobs,
                                });

                        state_choice.text.push_str(
                            choice
                                .delta
                                .content
                                .clone()
                                .unwrap_or("".to_string())
                                .as_str(),
                        );

                        if let Some(finish_reason) = choice.finish_reason {
                            state_choice.finish_reason = Some(finish_reason);
                        }
                    }
                }
                aggregator
            })
            .await;

        // If we have an error, return it
        let aggregator = if let Some(error) = aggregator.error {
            return Err(error);
        } else {
            aggregator
        };

        // extra the aggregated deltas and sort by index
        let mut choices: Vec<_> = aggregator
            .choices
            .into_values()
            .map(ChatCompletionChoice::from)
            .collect();

        choices.sort_by(|a, b| a.index.cmp(&b.index));

        Ok(ChatCompletionResponse {
            id: aggregator.id,
            created: aggregator.created,
            usage: aggregator.usage,
            model: aggregator.model,
            object: "chat.completion".to_string(),
            system_fingerprint: aggregator.system_fingerprint,
            choices,
            service_tier: aggregator.service_tier,
        })
    }
}

// todo - handle tool calls
impl From<DeltaChoice> for ChatCompletionChoice {
    fn from(delta: DeltaChoice) -> Self {
        ChatCompletionChoice {
            message: ChatCompletionContent {
                role: delta.role,
                content: Some(delta.text),
                tool_calls: None,
            },
            index: delta.index,
            finish_reason: delta.finish_reason.unwrap_or(FinishReason::length),
            logprobs: delta.logprobs,
        }
    }
}

impl ChatCompletionResponse {
    pub async fn from_sse_stream(
        stream: DataStream<Result<Message, SseCodecError>>,
    ) -> Result<ChatCompletionResponse, String> {
        let stream = convert_sse_stream::<ChatCompletionResponseDelta>(stream);
        ChatCompletionResponse::from_annotated_stream(stream).await
    }

    pub async fn from_annotated_stream(
        stream: DataStream<Annotated<ChatCompletionResponseDelta>>,
    ) -> Result<ChatCompletionResponse, String> {
        DeltaAggregator::apply(stream).await
    }
}

#[cfg(test)]
mod tests {
    use crate::protocols::openai::chat_completions::ChatCompletionChoiceDelta;

    use super::*;
    use futures::stream;

    fn create_test_delta(
        index: u64,
        text: &str,
        role: Option<MessageRole>,
        finish_reason: Option<FinishReason>,
    ) -> Annotated<ChatCompletionResponseDelta> {
        Annotated {
            data: Some(ChatCompletionResponseDelta {
                id: "test_id".to_string(),
                model: "meta/llama-3.1-8b-instruct".to_string(),
                created: 1234567890,
                service_tier: None,
                usage: None,
                system_fingerprint: None,
                choices: vec![ChatCompletionChoiceDelta {
                    index,
                    delta: ChatCompletionContent {
                        role,
                        content: Some(text.to_string()),
                        tool_calls: None,
                    },
                    finish_reason,
                    logprobs: None,
                }],
                object: "chat.completion".to_string(),
            }),
            id: Some("test_id".to_string()),
            event: None,
            comment: None,
        }
    }

    #[tokio::test]
    async fn test_empty_stream() {
        // Create an empty stream
        let stream: DataStream<Annotated<ChatCompletionResponseDelta>> = Box::pin(stream::empty());

        // Call DeltaAggregator::apply
        let result = DeltaAggregator::apply(stream).await;

        // Check the result
        assert!(result.is_ok());
        let response = result.unwrap();

        // Verify that the response is empty and has default values
        assert_eq!(response.id, "");
        assert_eq!(response.model, "");
        assert_eq!(response.created, 0);
        assert!(response.usage.is_none());
        assert!(response.system_fingerprint.is_none());
        assert_eq!(response.choices.len(), 0);
        assert!(response.service_tier.is_none());
    }

    #[tokio::test]
    async fn test_single_delta() {
        // Create a sample delta
        let annotated_delta = create_test_delta(0, "Hello,", Some(MessageRole::user), None);

        // Create a stream
        let stream = Box::pin(stream::iter(vec![annotated_delta]));

        // Call DeltaAggregator::apply
        let result = DeltaAggregator::apply(stream).await;

        // Check the result
        assert!(result.is_ok());
        let response = result.unwrap();

        // Verify the response fields
        assert_eq!(response.id, "test_id");
        assert_eq!(response.model, "meta/llama-3.1-8b-instruct");
        assert_eq!(response.created, 1234567890);
        assert!(response.usage.is_none());
        assert!(response.system_fingerprint.is_none());
        assert_eq!(response.choices.len(), 1);
        let choice = &response.choices[0];
        assert_eq!(choice.index, 0);
        assert_eq!(choice.message.content.as_ref().unwrap(), "Hello,");
        assert_eq!(choice.finish_reason, FinishReason::length);
        assert_eq!(choice.message.role.as_ref().unwrap(), &MessageRole::user);
        assert!(response.service_tier.is_none());
    }

    #[tokio::test]
    async fn test_multiple_deltas_same_choice() {
        // Create multiple deltas with the same choice index
        // One will have a MessageRole and no FinishReason,
        // the other will have a FinishReason and no MessageRole
        let annotated_delta1 = create_test_delta(0, "Hello,", Some(MessageRole::user), None);
        let annotated_delta2 = create_test_delta(0, " world!", None, Some(FinishReason::stop));

        // Create a stream
        let annotated_deltas = vec![annotated_delta1, annotated_delta2];
        let stream = Box::pin(stream::iter(annotated_deltas));

        // Call DeltaAggregator::apply
        let result = DeltaAggregator::apply(stream).await;

        // Check the result
        assert!(result.is_ok());
        let response = result.unwrap();

        // Verify the response fields
        assert_eq!(response.choices.len(), 1);
        let choice = &response.choices[0];
        assert_eq!(choice.index, 0);
        assert_eq!(choice.message.content.as_ref().unwrap(), "Hello, world!");
        assert_eq!(choice.finish_reason, FinishReason::stop);
        assert_eq!(choice.message.role.as_ref().unwrap(), &MessageRole::user);
    }

    #[tokio::test]
    async fn test_multiple_choices() {
        // Create a delta with multiple choices
        let delta = ChatCompletionResponseDelta {
            id: "test_id".to_string(),
            model: "test_model".to_string(),
            created: 1234567890,
            usage: None,
            system_fingerprint: None,
            choices: vec![
                ChatCompletionChoiceDelta {
                    index: 0,
                    delta: ChatCompletionContent {
                        role: Some(MessageRole::assistant),
                        content: Some("Choice 0".to_string()),
                        tool_calls: None,
                    },
                    finish_reason: Some(FinishReason::stop),
                    logprobs: None,
                },
                ChatCompletionChoiceDelta {
                    index: 1,
                    delta: ChatCompletionContent {
                        role: Some(MessageRole::assistant),
                        content: Some("Choice 1".to_string()),
                        tool_calls: None,
                    },
                    finish_reason: Some(FinishReason::stop),
                    logprobs: None,
                },
            ],
            object: "chat.completion".to_string(),
            service_tier: None,
        };

        // Wrap it in Annotated and create a stream
        let annotated_delta = Annotated {
            data: Some(delta),
            id: Some("test_id".to_string()),
            event: None,
            comment: None,
        };
        let stream = Box::pin(stream::iter(vec![annotated_delta]));

        // Call DeltaAggregator::apply
        let result = DeltaAggregator::apply(stream).await;

        // Check the result
        assert!(result.is_ok());
        let mut response = result.unwrap();

        // Verify the response fields
        assert_eq!(response.choices.len(), 2);
        response.choices.sort_by(|a, b| a.index.cmp(&b.index)); // Ensure the choices are ordered
        let choice0 = &response.choices[0];
        assert_eq!(choice0.index, 0);
        assert_eq!(choice0.message.content.as_ref().unwrap(), "Choice 0");
        assert_eq!(choice0.finish_reason, FinishReason::stop);
        assert_eq!(
            choice0.message.role.as_ref().unwrap(),
            &MessageRole::assistant
        );

        let choice1 = &response.choices[1];
        assert_eq!(choice1.index, 1);
        assert_eq!(choice1.message.content.as_ref().unwrap(), "Choice 1");
        assert_eq!(choice1.finish_reason, FinishReason::stop);
        assert_eq!(
            choice1.message.role.as_ref().unwrap(),
            &MessageRole::assistant
        );
    }
}
