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
use derive_builder::Builder;
use serde::de::{self, SeqAccess, Visitor};
use serde::ser::SerializeMap;
use serde::{Deserialize, Serialize};
use serde::{Deserializer, Serializer};
use std::collections::HashMap;
use std::fmt;
use std::fmt::Display;
use triton_distributed_runtime::protocols::annotated::AnnotationsProvider;
use validator::Validate;

mod aggregator;
mod delta;

pub use super::{CompletionTokensDetails, CompletionUsage, PromptTokensDetails};
// pub use aggregator::DeltaAggregator;
pub use delta::DeltaGenerator;

use super::{common::ChatCompletionLogprobs, ContentProvider};

/// Request object which is used to generate chat completions.
#[derive(Serialize, Deserialize, Validate, Debug, Clone)]
// #[builder(build_fn(private, name = "build_internal", validate = "Self::validate"))]
pub struct ChatCompletionRequest {
    #[serde(flatten)]
    pub chat_completion_request: async_openai::types::CreateChatCompletionRequest,
    pub nvext: Option<NvExt>,
}

/// Each turn in a conversation is represented by a ChatCompletionMessage.
#[derive(Builder, Debug, Deserialize, Serialize, Clone)]
pub struct ChatCompletionMessage {
    pub role: MessageRole,

    #[serde(deserialize_with = "deserialize_content")]
    pub content: Content,

    #[serde(skip_serializing_if = "Option::is_none", default)]
    #[builder(default)]
    pub name: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum MessageRole {
    user,
    system,
    assistant,
    function,
}

impl Display for MessageRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        use MessageRole::*;
        let s = match self {
            user => "user",
            system => "system",
            assistant => "assistant",
            function => "function",
        };
        write!(f, "{s}")
    }
}

#[derive(Debug, Deserialize, Clone, PartialEq, Eq)]
pub enum Content {
    Text(String),
    ImageUrl(Vec<ImageUrl>),
}

impl serde::Serialize for Content {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match *self {
            Content::Text(ref text) => serializer.serialize_str(text),
            Content::ImageUrl(ref image_url) => image_url.serialize(serializer),
        }
    }
}

fn deserialize_content<'de, D>(deserializer: D) -> Result<Content, D::Error>
where
    D: Deserializer<'de>,
{
    struct ContentVisitor;

    impl<'de> Visitor<'de> for ContentVisitor {
        type Value = Content;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("a string or an array of content parts")
        }

        fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(Content::Text(value.to_owned()))
        }

        fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
        where
            A: SeqAccess<'de>,
        {
            let mut parts = Vec::new();
            while let Some(value) = seq.next_element::<String>()? {
                if value.starts_with("http://") || value.starts_with("https://") {
                    parts.push(ImageUrl {
                        r#type: ContentType::image_url,
                        text: None,
                        image_url: Some(ImageUrlType { url: value }),
                    });
                } else {
                    parts.push(ImageUrl {
                        r#type: ContentType::text,
                        text: Some(value),
                        image_url: None,
                    });
                }
            }
            Ok(Content::ImageUrl(parts))
        }
    }

    deserializer.deserialize_any(ContentVisitor)
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum ContentType {
    text,
    image_url,
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub struct ImageUrlType {
    pub url: String,
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub struct ImageUrl {
    pub r#type: ContentType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_url: Option<ImageUrlType>,
}

/// Represents a chat completion response returned by model, based on the provided input.
pub type ChatCompletionResponse = ChatCompletionGeneric<ChatCompletionChoice>;

/// Represents a streamed chunk of a chat completion response returned by model, based on the provided input.
pub type ChatCompletionResponseDelta = ChatCompletionGeneric<ChatCompletionChoiceDelta>;

/// Common structure for chat completion responses; the only delta is the type of choices which differs
/// between streaming and non-streaming requests.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ChatCompletionGeneric<C>
where
    C: Serialize + Clone + ContentProvider,
{
    /// A unique identifier for the chat completion.
    pub id: String,

    /// A list of chat completion choices. Can be more than one if n is greater than 1.
    pub choices: Vec<C>,

    /// The Unix timestamp (in seconds) of when the chat completion was created.
    pub created: u64,

    /// The model used for the chat completion.
    pub model: String,

    /// The object type, which is `chat.completion` if the type of `Choice` is `ChatCompletionChoice`,
    /// or is `chat.completion.chunk` if the type of `Choice` is `ChatCompletionChoiceDelta`.
    pub object: String,

    /// Usage information for the completion request.
    pub usage: Option<CompletionUsage>,

    /// The service tier used for processing the request, optional.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<ServiceTier>,

    /// This fingerprint represents the backend configuration that the model runs with.
    ///
    /// Can be used in conjunction with the seed request parameter to understand when backend changes
    /// have been made that might impact determinism.
    ///
    /// NIM Compatibility:
    /// This field is not supported by the NIM; however it will be added in the future.
    /// The optional nature of this field will be relaxed when it is supported.
    pub system_fingerprint: Option<String>,
    // TODO() - add NvResponseExtention
}

// Enum for service tier, either "scale" or "default"
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "snake_case")]
pub enum ServiceTier {
    Auto,
    Scale,
    Default,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct ChatCompletionChoice {
    /// A chat completion message generated by the model.
    pub message: ChatCompletionContent,

    /// The index of the choice in the list of choices.
    pub index: u64,

    /// The reason the model stopped generating tokens. This will be `stop` if the model hit a natural
    /// stop point or a provided stop sequence, `length` if the maximum number of tokens specified
    /// in the request was reached, `content_filter` if content was omitted due to a flag from our content
    /// filters, `tool_calls` if the model called a tool, or `function_call` (deprecated) if the model called
    /// a function.
    ///
    /// NIM Compatibility:
    /// Only `stop` and `length` are currently supported by NIM.
    /// NIM may also provide additional reasons in the future, such as `error`, `timeout` or `cancelation`.
    pub finish_reason: FinishReason,

    /// Log probability information for the choice, optional field.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ChatCompletionLogprobs>,
}

impl ContentProvider for ChatCompletionChoice {
    fn content(&self) -> String {
        self.message.content()
    }
}

/// Same as ChatCompletionMessage, but received during a response stream.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatCompletionChoiceDelta {
    /// The index of the choice in the list of choices.
    pub index: u64,

    /// The reason the model stopped generating tokens. This will be `stop` if the model hit a natural
    /// stop point or a provided stop sequence, `length` if the maximum number of tokens specified
    /// in the request was reached, `content_filter` if content was omitted due to a flag from our content
    /// filters, `tool_calls` if the model called a tool, or `function_call` (deprecated) if the model called
    /// a function.
    ///
    /// NIM Compatibility:
    /// Only `stop` and `length` are currently supported by NIM.
    /// NIM may also provide additional reasons in the future, such as `error`, `timeout` or `cancelation`.
    pub finish_reason: Option<FinishReason>,

    /// A chat completion delta generated by streamed model responses.
    pub delta: ChatCompletionContent,

    /// Log probability information for the choice, optional field.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ChatCompletionLogprobs>,
}

impl ContentProvider for ChatCompletionChoiceDelta {
    fn content(&self) -> String {
        self.delta.content()
    }
}

/// A chat completion message generated by the model.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ChatCompletionContent {
    /// The role of the author of this message.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<MessageRole>,

    /// The contents of the message.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,

    /// Tool calls made by the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

impl ContentProvider for ChatCompletionContent {
    fn content(&self) -> String {
        self.content.clone().unwrap_or("".to_string())
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub enum ToolChoiceType {
    None,
    Auto,
    ToolChoice { tool: Tool },
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Eq)]
pub struct Function {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: FunctionParameters,
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum JSONSchemaType {
    Object,
    Number,
    String,
    Array,
    Null,
    Boolean,
}

#[derive(Debug, Deserialize, Serialize, Clone, Default, PartialEq, Eq)]
pub struct JSONSchemaDefine {
    #[serde(rename = "type")]
    pub schema_type: Option<JSONSchemaType>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enum_values: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub properties: Option<HashMap<String, Box<JSONSchemaDefine>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub items: Option<Box<JSONSchemaDefine>>,
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Eq)]
pub struct FunctionParameters {
    #[serde(rename = "type")]
    pub schema_type: JSONSchemaType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub properties: Option<HashMap<String, Box<JSONSchemaDefine>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required: Option<Vec<String>>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum FinishReason {
    stop,
    length,
    content_filter,
    tool_calls,
    cancelled,
    null,
}

/// from_str trait
impl std::str::FromStr for FinishReason {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "stop" => Ok(FinishReason::stop),
            "length" => Ok(FinishReason::length),
            "content_filter" => Ok(FinishReason::content_filter),
            "tool_calls" => Ok(FinishReason::tool_calls),
            "null" => Ok(FinishReason::null),
            _ => Err(format!("Unknown FinishReason: {}", s)),
        }
    }
}

impl std::fmt::Display for FinishReason {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            FinishReason::stop => write!(f, "stop"),
            FinishReason::length => write!(f, "length"),
            FinishReason::content_filter => write!(f, "content_filter"),
            FinishReason::tool_calls => write!(f, "tool_calls"),
            FinishReason::cancelled => write!(f, "cancelled"),
            FinishReason::null => write!(f, "null"),
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[allow(non_camel_case_types)]
pub struct FinishDetails {
    pub r#type: FinishReason,
    pub stop: String,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ToolCall {
    pub id: String,
    pub r#type: String,
    pub function: ToolCallFunction,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ToolCallFunction {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

#[allow(dead_code)]
fn serialize_tool_choice<S>(
    value: &Option<ToolChoiceType>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    match value {
        Some(ToolChoiceType::None) => serializer.serialize_str("none"),
        Some(ToolChoiceType::Auto) => serializer.serialize_str("auto"),
        Some(ToolChoiceType::ToolChoice { tool }) => {
            let mut map = serializer.serialize_map(Some(2))?;
            map.serialize_entry("type", &tool.r#type)?;
            map.serialize_entry("function", &tool.function)?;
            map.end()
        }
        None => serializer.serialize_none(),
    }
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Eq)]
pub struct Tool {
    pub r#type: ToolType,
    pub function: Function,
}

#[derive(Debug, Deserialize, Serialize, Copy, Clone, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ToolType {
    Function,
}

// impl ChatCompletionRequest {}

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
        self.chat_completion_request.temperature
    }

    fn get_top_p(&self) -> Option<f32> {
        self.chat_completion_request.top_p
    }

    fn get_frequency_penalty(&self) -> Option<f32> {
        self.chat_completion_request.frequency_penalty
    }

    fn get_presence_penalty(&self) -> Option<f32> {
        self.chat_completion_request.presence_penalty
    }

    fn nvext(&self) -> Option<&NvExt> {
        self.nvext.as_ref()
    }
}

impl OpenAIStopConditionsProvider for ChatCompletionRequest {
    fn get_max_tokens(&self) -> Option<i32> {
        // TODO THIS IS WRONG i32 -> u32
        // self.chat_completion_request.max_tokens
        None
    }

    fn get_min_tokens(&self) -> Option<i32> {
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
