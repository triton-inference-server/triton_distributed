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

use super::{ChatCompletionResponse, ChatCompletionResponseDelta};
use crate::protocols::{
    codec::{Message, SseCodecError},
    convert_sse_stream, Annotated,
};

use futures::{Stream, StreamExt};
use std::{collections::HashMap, pin::Pin};

type DataStream<T> = Pin<Box<dyn Stream<Item = T> + Send + Sync>>;

/// Aggregates a stream of [`ChatCompletionResponseDelta`]s into a single [`ChatCompletionResponse`].
pub struct DeltaAggregator {
    id: String,
    model: String,
    created: u32,
    usage: Option<async_openai::types::CompletionUsage>,
    system_fingerprint: Option<String>,
    choices: HashMap<u32, DeltaChoice>,
    error: Option<String>,
    service_tier: Option<async_openai::types::ServiceTierResponse>,
}

// Holds the accumulated state of a choice
struct DeltaChoice {
    index: u32,
    text: String,
    role: Option<async_openai::types::Role>,
    finish_reason: Option<async_openai::types::FinishReason>,
    logprobs: Option<async_openai::types::ChatChoiceLogprobs>,
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
                    aggregator.id = delta.inner.id;
                    aggregator.model = delta.inner.model;
                    aggregator.created = delta.inner.created;
                    aggregator.service_tier = delta.inner.service_tier;
                    if let Some(usage) = delta.inner.usage {
                        aggregator.usage = Some(usage);
                    }
                    if let Some(system_fingerprint) = delta.inner.system_fingerprint {
                        aggregator.system_fingerprint = Some(system_fingerprint);
                    }

                    // handle the choices
                    for choice in delta.inner.choices {
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

                        if let Some(content) = &choice.delta.content {
                            state_choice.text.push_str(content);
                        }

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
            .map(async_openai::types::ChatChoice::from)
            .collect();

        choices.sort_by(|a, b| a.index.cmp(&b.index));

        let inner = async_openai::types::CreateChatCompletionResponse {
            id: aggregator.id,
            created: aggregator.created,
            usage: aggregator.usage,
            model: aggregator.model,
            object: "chat.completion".to_string(),
            system_fingerprint: aggregator.system_fingerprint,
            choices,
            service_tier: aggregator.service_tier,
        };

        let response = ChatCompletionResponse { inner };

        Ok(response)
    }
}

// todo - handle tool calls
#[allow(deprecated)]
impl From<DeltaChoice> for async_openai::types::ChatChoice {
    fn from(delta: DeltaChoice) -> Self {
        // ALLOW: function_call is deprecated
        async_openai::types::ChatChoice {
            message: async_openai::types::ChatCompletionResponseMessage {
                role: delta.role.expect("delta should have a Role"),
                content: Some(delta.text),
                tool_calls: None,
                refusal: None,
                function_call: None,
                audio: None,
            },
            index: delta.index,
            finish_reason: delta.finish_reason,
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

    use super::*;
    use futures::stream;

    #[allow(deprecated)]
    fn create_test_delta(
        index: u32,
        text: &str,
        role: Option<async_openai::types::Role>,
        finish_reason: Option<async_openai::types::FinishReason>,
    ) -> Annotated<ChatCompletionResponseDelta> {
        // ALLOW: function_call is deprecated
        let delta = async_openai::types::ChatCompletionStreamResponseDelta {
            content: Some(text.to_string()),
            function_call: None,
            tool_calls: None,
            role,
            refusal: None,
        };
        let choice = async_openai::types::ChatChoiceStream {
            index,
            delta,
            finish_reason,
            logprobs: None,
        };

        let inner = async_openai::types::CreateChatCompletionStreamResponse {
            id: "test_id".to_string(),
            model: "meta/llama-3.1-8b-instruct".to_string(),
            created: 1234567890,
            service_tier: None,
            usage: None,
            system_fingerprint: None,
            choices: vec![choice],
            object: "chat.completion".to_string(),
        };

        let data = ChatCompletionResponseDelta { inner };

        Annotated {
            data: Some(data),
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
        assert_eq!(response.inner.id, "");
        assert_eq!(response.inner.model, "");
        assert_eq!(response.inner.created, 0);
        assert!(response.inner.usage.is_none());
        assert!(response.inner.system_fingerprint.is_none());
        assert_eq!(response.inner.choices.len(), 0);
        assert!(response.inner.service_tier.is_none());
    }

    #[tokio::test]
    async fn test_single_delta() {
        // Create a sample delta
        let annotated_delta =
            create_test_delta(0, "Hello,", Some(async_openai::types::Role::User), None);

        // Create a stream
        let stream = Box::pin(stream::iter(vec![annotated_delta]));

        // Call DeltaAggregator::apply
        let result = DeltaAggregator::apply(stream).await;

        // Check the result
        assert!(result.is_ok());
        let response = result.unwrap();

        // Verify the response fields
        assert_eq!(response.inner.id, "test_id");
        assert_eq!(response.inner.model, "meta/llama-3.1-8b-instruct");
        assert_eq!(response.inner.created, 1234567890);
        assert!(response.inner.usage.is_none());
        assert!(response.inner.system_fingerprint.is_none());
        assert_eq!(response.inner.choices.len(), 1);
        let choice = &response.inner.choices[0];
        assert_eq!(choice.index, 0);
        assert_eq!(choice.message.content.as_ref().unwrap(), "Hello,");
        assert!(choice.finish_reason.is_none());
        assert_eq!(choice.message.role, async_openai::types::Role::User);
        assert!(response.inner.service_tier.is_none());
    }

    #[tokio::test]
    async fn test_multiple_deltas_same_choice() {
        // Create multiple deltas with the same choice index
        // One will have a MessageRole and no FinishReason,
        // the other will have a FinishReason and no MessageRole
        let annotated_delta1 =
            create_test_delta(0, "Hello,", Some(async_openai::types::Role::User), None);
        let annotated_delta2 = create_test_delta(
            0,
            " world!",
            None,
            Some(async_openai::types::FinishReason::Stop),
        );

        // Create a stream
        let annotated_deltas = vec![annotated_delta1, annotated_delta2];
        let stream = Box::pin(stream::iter(annotated_deltas));

        // Call DeltaAggregator::apply
        let result = DeltaAggregator::apply(stream).await;

        // Check the result
        assert!(result.is_ok());
        let response = result.unwrap();

        // Verify the response fields
        assert_eq!(response.inner.choices.len(), 1);
        let choice = &response.inner.choices[0];
        assert_eq!(choice.index, 0);
        assert_eq!(choice.message.content.as_ref().unwrap(), "Hello, world!");
        assert_eq!(
            choice.finish_reason,
            Some(async_openai::types::FinishReason::Stop)
        );
        assert_eq!(choice.message.role, async_openai::types::Role::User);
    }

    #[allow(deprecated)]
    #[tokio::test]
    async fn test_multiple_choices() {
        // Create a delta with multiple choices
        // ALLOW: function_call is deprecated
        let delta = async_openai::types::CreateChatCompletionStreamResponse {
            id: "test_id".to_string(),
            model: "test_model".to_string(),
            created: 1234567890,
            service_tier: None,
            usage: None,
            system_fingerprint: None,
            choices: vec![
                async_openai::types::ChatChoiceStream {
                    index: 0,
                    delta: async_openai::types::ChatCompletionStreamResponseDelta {
                        role: Some(async_openai::types::Role::Assistant),
                        content: Some("Choice 0".to_string()),
                        function_call: None,
                        tool_calls: None,
                        refusal: None,
                    },
                    finish_reason: Some(async_openai::types::FinishReason::Stop),
                    logprobs: None,
                },
                async_openai::types::ChatChoiceStream {
                    index: 1,
                    delta: async_openai::types::ChatCompletionStreamResponseDelta {
                        role: Some(async_openai::types::Role::Assistant),
                        content: Some("Choice 1".to_string()),
                        function_call: None,
                        tool_calls: None,
                        refusal: None,
                    },
                    finish_reason: Some(async_openai::types::FinishReason::Stop),
                    logprobs: None,
                },
            ],
            object: "chat.completion".to_string(),
        };

        let data = ChatCompletionResponseDelta { inner: delta };

        // Wrap it in Annotated and create a stream
        let annotated_delta = Annotated {
            data: Some(data),
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
        assert_eq!(response.inner.choices.len(), 2);
        response.inner.choices.sort_by(|a, b| a.index.cmp(&b.index)); // Ensure the choices are ordered
        let choice0 = &response.inner.choices[0];
        assert_eq!(choice0.index, 0);
        assert_eq!(choice0.message.content.as_ref().unwrap(), "Choice 0");
        assert_eq!(
            choice0.finish_reason,
            Some(async_openai::types::FinishReason::Stop)
        );
        assert_eq!(choice0.message.role, async_openai::types::Role::Assistant);

        let choice1 = &response.inner.choices[1];
        assert_eq!(choice1.index, 1);
        assert_eq!(choice1.message.content.as_ref().unwrap(), "Choice 1");
        assert_eq!(
            choice1.finish_reason,
            Some(async_openai::types::FinishReason::Stop)
        );
        assert_eq!(choice1.message.role, async_openai::types::Role::Assistant);
    }
}
