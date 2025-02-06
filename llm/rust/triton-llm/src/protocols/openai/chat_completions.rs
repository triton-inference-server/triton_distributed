// Copyright 2024-2025 NVIDIA CORPORATION & AFFILIATES
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

use std::collections::HashMap;
use std::collections::VecDeque;
use std::fmt;
use std::fmt::Display;

use derive_builder::Builder;
use serde::de::{self, SeqAccess, Visitor};
use serde::ser::SerializeMap;
use serde::{Deserialize, Serialize};
use serde::{Deserializer, Serializer};
use serde_json::Value;
use validator::Validate;

mod aggregator;
mod delta;

use super::nvext::NvExtProvider;
pub use super::{CompletionTokensDetails, CompletionUsage, PromptTokensDetails};
// pub use aggregator::DeltaAggregator;
pub use delta::DeltaGenerator;

use super::{
    common::{self, ChatCompletionLogprobs, SamplingOptionsProvider, StopConditionsProvider},
    nvext::NvExt,
    validate_logit_bias, ContentProvider, OpenAISamplingOptionsProvider,
    OpenAIStopConditionsProvider,
};
// use crate::AnnotationsProvider;

/// Request object which is used to generate chat completions.
#[derive(Serialize, Deserialize, Builder, Validate, Debug, Clone)]
#[builder(build_fn(private, name = "build_internal", validate = "Self::validate"))]
pub struct ChatCompletionRequest {
    /// Multi-turn chat messages.
    ///
    /// NIM Compatibility:
    /// Multi-turn chat models vary, some of which work with the OpenAI ChatGPT format, while others
    /// will require `NvExt`.
    pub messages: Vec<ChatCompletionMessage>,

    /// Name of the model
    #[builder(setter(into))]
    pub model: String,

    /// The maximum number of tokens that can be generated in the completion.
    /// The token count of your prompt plus max_tokens cannot exceed the model's context length.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into, strip_option))]
    #[validate(range(min = 1))]
    pub max_tokens: Option<i32>,

    /// The minimum number of tokens to generate. We ignore stop tokens until we see this many
    /// tokens. Leave this None unless you are working on the pre-processor.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into, strip_option))]
    pub min_tokens: Option<i32>,

    /// If set, partial message deltas will be sent, like in ChatGPT. Tokens will be sent as data-only
    /// server-sent events as they become available, with the stream terminated by a data: \[DONE\]
    ///
    /// NIM Compatibility:
    /// The NIM SDK can send extra meta data in the SSE stream using the `:` comment, `event:`,
    /// or `id:` fields. See the `enable_sse_metadata` field in the NvExt object.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub stream: Option<bool>,

    /// How many chat completion choices to generate for each input message.
    ///
    /// NIM Compatibility:
    /// Values greater than 1 are not currently supported by NIM.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into, strip_option))]
    pub n: Option<i32>,

    /// What sampling `temperature` to use, between 0 and 2. Higher values like 0.8 will make the
    /// output more random, while lower values like 0.2 will make it more focused and deterministic.
    /// OpenAI defaults to 1.0; however, in this crate, the default is None, and model-specific defaults
    /// can be applied later as part of associating the request with a given model.
    ///
    /// OpenAI generally recommend altering this or `top_p` but not both.
    ///
    /// TODO(): Add a model specific validation which could enforce only a single type of sampling can be used.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(range(min = "super::MIN_TEMPERATURE", max = "super::MAX_TEMPERATURE"))]
    #[builder(default, setter(into, strip_option))]
    pub temperature: Option<f32>,

    /// An alternative to sampling with `temperature`, called nucleus sampling, where the model
    /// considers the results of the tokens with `top_p` probability mass. So 0.1 means only the tokens
    /// comprising the top 10% probability mass are considered.
    ///
    /// We generally recommend altering this or `temperature` but not both.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(range(min = "super::MIN_TOP_P", max = "super::MAX_TOP_P"))]
    #[builder(default, setter(into, strip_option))]
    pub top_p: Option<f32>,

    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency
    /// in the text so far, decreasing the model's likelihood to repeat the same line verbatim.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(range(
        min = "super::MIN_FREQUENCY_PENALTY",
        max = "super::MAX_FREQUENCY_PENALTY"
    ))]
    #[builder(default, setter(into, strip_option))]
    pub frequency_penalty: Option<f32>,

    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in
    /// the text so far, increasing the model's likelihood to talk about new topics.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(range(
        min = "super::MIN_PRESENCE_PENALTY",
        max = "super::MAX_PRESENCE_PENALTY"
    ))]
    #[builder(default, setter(into, strip_option))]
    pub presence_penalty: Option<f32>,

    /// OpenAI specific API fields:
    /// See: <https://platform.openai.com/docs/api-reference/chat/create#chat-create-response_format>
    ///
    /// NIM Compatibility:
    /// This option is not currently supported by NIM LLM. An error will be returned if this field is set.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default)]
    pub response_format: Option<Value>,

    /// Up to 4 sequences where the API will stop generating further tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(length(max = 4))]
    #[builder(default, setter(into, strip_option))]
    pub stop: Option<Vec<String>>,

    /// Whether to return log probabilities of the output tokens or not. If true, returns the log probabilities
    /// of each output token returned in the content of message.
    ///
    /// Not all models support logprobs. If logprobs is set to true for a model that does not support it,
    /// the request will be processed as if logprobs is set to false.
    ///
    /// NIM Compatibility:
    /// TODO - Add a NvExt `strict` object which will disable relaxing of model specific limitations; meaning,
    /// if the user requests `logprobs` and the model does not support them, the request will fail wth an error.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub logprobs: Option<bool>,

    /// An integer between 0 and 20 specifying the number of most likely tokens to return at each token position,
    /// each with an associated log probability. logprobs must be set to true if this parameter is used.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(range(min = 0, max = 20))]
    #[builder(default, setter(into, strip_option))]
    pub top_logprobs: Option<i32>,

    /// Modify the likelihood of specified tokens appearing in the completion.
    ///
    /// Accepts a JSON object that maps tokens (specified by their token ID in the GPT tokenizer) to an
    /// associated bias value from -100 to 100. You can use this tokenizer tool to convert text to token IDs.
    /// Mathematically, the bias is added to the logits generated by the model prior to sampling. The exact
    /// effect will vary per model, but values between -1 and 1 should decrease or increase likelihood of
    /// selection; values like -100 or 100 should result in a ban or exclusive selection of the relevant token.
    ///
    /// As specified in the OpenAI examples, this is a map of tokens_ids as strings to a bias value that
    /// is an integer.
    ///
    /// However, the OpenAI blog using the SDK shows that it can also be specified more accurately as a
    /// map of token_ids as ints to a bias value that is also an int.
    ///
    /// NIM Compatibility:
    /// In the conversion of the OpenAI request to the internal NIM format, the keys of this map will be
    /// validated to ensure they are integers. Since different models may have different tokenizers, the
    /// range and values will again be validated on the compute backend to ensure they map to valid tokens
    /// in the vocabulary of the model.
    ///
    /// ```
    /// use triton_llm::protocols::openai::completions::CompletionRequest;
    ///
    /// let request = CompletionRequest::builder()
    ///     .prompt("What is the meaning of life?")
    ///     .model("meta/llama-3.1-8b-instruct")
    ///     .add_logit_bias(1337, -100) // using an int as a key is ok
    ///     .add_logit_bias("42", 100)  // using a string as a key is also ok
    ///     .build()
    ///     .expect("Should not fail");
    ///
    /// assert!(CompletionRequest::builder()
    ///     .prompt("What is the meaning of life?")
    ///     .model("meta/llama-3.1-8b-instruct")
    ///     .add_logit_bias("some non int", -100)
    ///     .build()
    ///     .is_err());
    /// ```
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(custom(function = "validate_logit_bias"))]
    #[builder(default, setter(into, strip_option))]
    pub logit_bias: Option<HashMap<String, i32>>,

    /// A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse.
    ///
    /// NIM Compatibility:
    /// If provided, then the value of this field will be included in the trace metadata and the accounting
    /// data (if enabled).
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into, strip_option))]
    pub user: Option<String>,

    /// If specified, our system will make a best effort to sample deterministically, such that repeated
    /// requests with the same seed and parameters should return the same result. Determinism is not guaranteed,
    /// and you should refer to the `system_fingerprint` response parameter to monitor changes in the backend.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into, strip_option))]
    pub seed: Option<i64>,

    /// A list of tools the model may call. Currently, only functions are supported as a tool. Use this to
    /// provide a list of functions the model may generate JSON inputs for. A max of 128 functions are supported.
    ///
    /// NIM Compatibility:
    /// This field is not currently supported by NIM LLM. An error will be returned if this field is set.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default)]
    pub tools: Option<Vec<Tool>>,

    /// Controls which (if any) function is called by the model. none means the model will not call a function
    /// and instead generates a message. auto means the model can pick between generating a message or calling
    /// a function. Specifying a particular function via {"type": "function", "function": {"name": "my_function"}}
    /// forces the model to call that function.
    ///
    /// `none` is the default when no functions are present. `auto` is the default if functions are present.
    ///
    /// NIM Compatibility:
    /// This field is not currently supported by NIM LLM. An error will be returned if this field is set.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(serialize_with = "serialize_tool_choice")]
    #[builder(default)]
    pub tool_choice: Option<ToolChoiceType>,

    /// Additional parameters supported by NIM backends
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub nvext: Option<NvExt>,
}

impl ChatCompletionRequest {
    pub fn builder() -> ChatCompletionRequestBuilder {
        ChatCompletionRequestBuilder::default()
    }
}

impl ChatCompletionRequestBuilder {
    // This is a pre-build validate function
    // This is called before the generated build method, in this case build_internal, is called
    // This has access to the internal state of the builder
    fn validate(&self) -> Result<(), String> {
        Ok(())
    }

    /// Builds and validates the ChatCompletionRequest
    ///
    /// ```rust
    /// use triton_llm::protocols::openai::chat_completions::ChatCompletionRequest;
    ///
    /// let request = ChatCompletionRequest::builder()
    ///     .model("mixtral-8x7b-instruct-v0.1")
    ///     .add_user_message("Hello")
    ///     .max_tokens(16)
    ///     .build()
    ///     .expect("Failed to build ChatCompletionRequest");
    /// ```
    pub fn build(&self) -> anyhow::Result<ChatCompletionRequest> {
        // Calls the build_private, validates the result, then performs addition
        // post build validation where we are looking a mutually exclusive fields
        // and ensuring that there are not mutually exclusive collisions.
        let request = self
            .build_internal()
            .map_err(|e| anyhow::anyhow!("Failed to build ChatCompletionRequest: {}", e))?;

        request
            .validate()
            .map_err(|e| anyhow::anyhow!("Failed to validate ChatCompletionRequest: {}", e))?;

        // check mutually exclusive fields
        if request.top_logprobs.is_some() {
            if request.logprobs.is_none() {
                anyhow::bail!("top_logprobs requires logprobs to be set to true");
            }
            if let Some(logprobs) = request.logprobs {
                if !logprobs {
                    anyhow::bail!("top_logprobs requires logprobs to be set to true");
                }
            }
        }

        Ok(request)
    }

    /// Add a message to the `Vec<ChatCompletionMessage>` in the ChatCompletionRequest
    /// This will either create or append to the `Vec<ChatCompletionMessage>`
    pub fn add_message(&mut self, message: ChatCompletionMessage) -> &mut Self {
        // If messages exist we get them or we create new messages with Vec::new
        self.messages.get_or_insert_with(Vec::new).push(message);
        self
    }

    /// Add a user message to the `Vec<ChatCompletionMessage>` in the ChatCompletionRequest
    pub fn add_user_message(&mut self, content: impl Into<String>) -> &mut Self {
        self.add_message(ChatCompletionMessage {
            role: MessageRole::user,
            content: Content::Text(content.into()),
            name: None,
        })
    }

    /// Add an assistant message to the `Vec<ChatCompletionMessage>` in the ChatCompletionRequest
    pub fn add_assistant_message(&mut self, content: impl Into<String>) -> &mut Self {
        self.add_message(ChatCompletionMessage {
            role: MessageRole::assistant,
            content: Content::Text(content.into()),
            name: None,
        })
    }

    /// Add a system message to the `Vec<ChatCompletionMessage>` in the ChatCompletionRequest
    pub fn add_system_message(&mut self, content: impl Into<String>) -> &mut Self {
        self.add_message(ChatCompletionMessage {
            role: MessageRole::system,
            content: Content::Text(content.into()),
            name: None,
        })
    }

    /// Add a stop condition to the `Vec<String>` in the ChatCompletionRequest
    /// This will either create or append to the `Vec<String>`
    pub fn add_stop(&mut self, stop: impl Into<String>) -> &mut Self {
        self.stop
            .get_or_insert_with(|| Some(vec![]))
            .as_mut()
            .expect("stop should always be Some(Vec)")
            .push(stop.into());
        self
    }

    /// Add a token and bias to the `HashMap<String, i32>` in the ChatCompletionRequest
    /// This will either create or update the `HashMap<String, i32>`
    /// See: [`ChatCompletionRequest::logit_bias`] for more details
    pub fn add_logit_bias<T>(&mut self, token_id: T, bias: i32) -> &mut Self
    where
        T: std::fmt::Display,
    {
        self.logit_bias
            .get_or_insert_with(|| Some(HashMap::new()))
            .as_mut()
            .expect("logit_bias should always be Some(HashMap)")
            .insert(token_id.to_string(), bias);

        self
    }
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

impl ChatCompletionRequest {}

impl NvExtProvider for ChatCompletionRequest {
    fn nvext(&self) -> Option<&NvExt> {
        self.nvext.as_ref()
    }

    fn raw_prompt(&self) -> Option<String> {
        None
    }
}

// impl AnnotationsProvider for ChatCompletionRequest {
//     fn annotations(&self) -> Option<Vec<String>> {
//         self.nvext
//             .as_ref()
//             .and_then(|nvext| nvext.annotations.clone())
//     }

//     fn has_annotation(&self, annotation: &str) -> bool {
//         self.nvext
//             .as_ref()
//             .and_then(|nvext| nvext.annotations.as_ref())
//             .map(|annotations| annotations.contains(&annotation.to_string()))
//             .unwrap_or(false)
//     }
// }

impl OpenAISamplingOptionsProvider for ChatCompletionRequest {
    fn get_temperature(&self) -> Option<f32> {
        self.temperature
    }

    fn get_top_p(&self) -> Option<f32> {
        self.top_p
    }

    fn get_frequency_penalty(&self) -> Option<f32> {
        self.frequency_penalty
    }

    fn get_presence_penalty(&self) -> Option<f32> {
        self.presence_penalty
    }

    fn nvext(&self) -> Option<&NvExt> {
        self.nvext.as_ref()
    }
}

impl OpenAIStopConditionsProvider for ChatCompletionRequest {
    fn get_max_tokens(&self) -> Option<i32> {
        self.max_tokens
    }

    fn get_min_tokens(&self) -> Option<i32> {
        self.min_tokens
    }

    fn get_stop(&self) -> Option<Vec<String>> {
        self.stop.clone()
    }

    fn nvext(&self) -> Option<&NvExt> {
        self.nvext.as_ref()
    }
}

/// Implements TryFrom for converting an OpenAI's ChatCompletionRequest to an Engine's CompletionRequest
impl TryFrom<ChatCompletionRequest> for common::CompletionRequest {
    type Error = anyhow::Error;

    fn try_from(request: ChatCompletionRequest) -> Result<Self, Self::Error> {
        // openai_api_rs::v1::chat_completion
        // pub struct ChatCompletionRequest {
        //  NA pub model: String,
        //  L  pub messages: Vec<ChatCompletionMessage, Global>,
        //  SO pub temperature: Option<f32>,
        //  SO pub top_p: Option<f32>,
        //  SO pub n: Option<i32>,
        //  ** pub response_format: Option<Value>,
        //  NA pub stream: Option<bool>,  // See Issue #8
        //  SC pub stop: Option<Vec<String, Global>>,
        //  SC pub max_tokens: Option<i32>,
        //  SO pub presence_penalty: Option<f32>,
        //  SO pub frequency_penalty: Option<f32>,
        //  ** pub logit_bias: Option<HashMap<String, i32, RandomState>>,
        //  ** pub user: Option<String>,
        //  SO pub seed: Option<i64>,
        //  ** pub tools: Option<Vec<Tool, Global>>,
        //  ** pub tool_choice: Option<ToolChoiceType>,
        // }
        //
        // ** not supported
        // NA not applicable
        // L  local in this method
        // SO extract_sampling_options
        // SC extract_stop_conditions

        // first we validate the OpenAI request
        // we can not validate everything as some fields require backend awareness
        // however, we can validate against the public OpenAI limit
        request
            .validate()
            .map_err(|e| anyhow::anyhow!("Failed to validate ChatCompletionRequest: {}", e))?;

        // todo(ryan) - open a ticket to support this
        if request.logit_bias.is_some() {
            anyhow::bail!("logit_bias is not supported");
        }

        // todo(ryan) - add support for user
        if request.user.is_some() {
            anyhow::bail!("user is not supported");
        }

        if request.response_format.is_some() {
            anyhow::bail!("response_format is not supported");
        }

        if request.tools.is_some() {
            anyhow::bail!("tools is not supported");
        }

        if request.tool_choice.is_some() {
            anyhow::bail!("tool_choice is not supported");
        }

        // sampling options
        let sampling_options = request
            .extract_sampling_options()
            .map_err(|e| anyhow::anyhow!("Failed to extract SamplingOptions: {}", e))?;

        // stop conditions
        let stop_conditions = request
            .extract_stop_conditions()
            .map_err(|e| anyhow::anyhow!("Failed to extract StopConditions: {}", e))?;

        // first we need to process the messages
        let prompt = common::PromptType::ChatCompletion(
            validate_and_collect_chat_messages(request.messages)
                .map_err(|e| anyhow::anyhow!("Failed to validate chat messages: {}", e))?,
        );

        // return the completion request
        Ok(common::CompletionRequest {
            prompt,
            stop_conditions,
            sampling_options,
            mdc_sum: None,
            annotations: None,
        })
    }
}

impl TryFrom<common::StreamingCompletionResponse> for ChatCompletionChoice {
    type Error = anyhow::Error;

    fn try_from(response: common::StreamingCompletionResponse) -> Result<Self, Self::Error> {
        let choice = ChatCompletionChoice {
            index: response.delta.index.unwrap_or(0) as u64,
            message: ChatCompletionContent {
                role: Some(MessageRole::assistant),
                content: response.delta.text,
                tool_calls: None,
            },

            finish_reason: match &response.delta.finish_reason {
                Some(common::FinishReason::EoS) => FinishReason::stop,
                Some(common::FinishReason::Stop) => FinishReason::stop,
                Some(common::FinishReason::Length) => FinishReason::length,
                Some(common::FinishReason::Error(err_msg)) => {
                    return Err(anyhow::anyhow!("finish_reason::error = {}", err_msg));
                }
                Some(common::FinishReason::Cancelled) => FinishReason::null,
                None => FinishReason::null,
            },

            logprobs: response.logprobs,
        };

        Ok(choice)
    }
}

impl TryFrom<common::StreamingCompletionResponse> for ChatCompletionChoiceDelta {
    type Error = anyhow::Error;

    fn try_from(response: common::StreamingCompletionResponse) -> Result<Self, Self::Error> {
        let choice = ChatCompletionChoiceDelta {
            index: response.delta.index.unwrap_or(0) as u64,
            delta: ChatCompletionContent {
                role: Some(MessageRole::assistant),
                content: response.delta.text,
                tool_calls: None,
            },

            finish_reason: match &response.delta.finish_reason {
                Some(common::FinishReason::EoS) => Some(FinishReason::stop),
                Some(common::FinishReason::Stop) => Some(FinishReason::stop),
                Some(common::FinishReason::Length) => Some(FinishReason::length),
                Some(common::FinishReason::Error(err_msg)) => {
                    return Err(anyhow::anyhow!("finish_reason::error = {}", err_msg));
                }
                Some(common::FinishReason::Cancelled) => Some(FinishReason::null),
                None => None,
            },
            logprobs: response.logprobs,
        };

        Ok(choice)
    }
}

fn validate_and_collect_chat_messages(
    messages: Vec<ChatCompletionMessage>,
) -> Result<common::ChatContext, anyhow::Error> {
    let mut system_prompt = None;
    let mut turns = VecDeque::new();
    let mut last_role = MessageRole::assistant;

    for message in messages {
        match message.role {
            MessageRole::system => {
                if system_prompt.is_some() {
                    return Err(anyhow::anyhow!("More than one system message found"));
                }
                system_prompt = Some(message.content);
            }
            MessageRole::user | MessageRole::assistant => {
                if last_role == message.role {
                    if turns.is_empty() {
                        return Err(anyhow::anyhow!("First message must be a user message"));
                    }
                    return Err(anyhow::anyhow!(
                        "User and assistant messages must alternate"
                    ));
                }
                last_role = message.role.clone();
                turns.push_back(message);
            }
            MessageRole::function => {} // Ignoring function messages as per assumption.
        }
    }

    if let Some(first) = turns.front() {
        if let MessageRole::assistant = first.role {
            return Err(anyhow::anyhow!("Sequence must start with a user message"));
        }
    }

    if turns.len() % 2 == 0 {
        return Err(anyhow::anyhow!("Sequence must end with a user message"));
    }

    let mut context = Vec::new();
    while turns.len() >= 2 {
        let user = turns.pop_front().unwrap();
        let asst = turns.pop_front().unwrap();

        let user = match user.content {
            Content::Text(text) => text,
            _ => return Err(anyhow::anyhow!("User message must be text")),
        };
        let asst = match asst.content {
            Content::Text(text) => text,
            _ => return Err(anyhow::anyhow!("Assistant message must be text")),
        };
        context.push(common::ChatTurn {
            user,
            assistant: asst,
        });
    }

    let prompt = turns.pop_back().unwrap();
    let prompt = match prompt.content {
        Content::Text(text) => text,
        _ => return Err(anyhow::anyhow!("Prompt message must be text")),
    };

    let system_prompt = match system_prompt {
        Some(Content::Text(text)) => Some(text),
        Some(_) => return Err(anyhow::anyhow!("System prompt must be text")),
        None => None,
    };

    Ok(common::ChatContext {
        completion: common::CompletionContext {
            prompt,
            system_prompt,
        },
        context,
    })
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use serde_json::json;
    use std::error::Error;

    use super::*;

    #[test]
    fn test_chat_completions_valid_request_minimal() -> Result<(), Box<dyn Error>> {
        let request = ChatCompletionRequest::builder()
            .model("meta/llama-3.1-8b-instruct")
            .add_user_message("Hello!")
            .build();

        assert!(
            request.is_ok(),
            "Request should succeed with minimal fields"
        );
        Ok(())
    }

    #[test]
    fn test_chat_completions_valid_request_full() -> Result<(), Box<dyn Error>> {
        let request = ChatCompletionRequest::builder()
            .model("meta/llama-3.1-8b-instruct")
            .add_user_message("Hello!")
            .max_tokens(50)
            .stream(true)
            .n(1)
            .temperature(1.0)
            .top_p(0.9)
            .frequency_penalty(0.5)
            .presence_penalty(0.5)
            .stop(vec!["The end.".to_string()])
            .logprobs(true)
            .top_logprobs(5)
            .logit_bias(HashMap::new())
            .user("test_user")
            .seed(1234)
            .build();

        println!("{:?}", request);

        assert!(
            request.is_ok(),
            "Request should succeed with all fields set"
        );
        Ok(())
    }

    #[test]
    fn test_chat_completions_top_logprobs_requires_logprobs() -> Result<(), Box<dyn Error>> {
        let request = ChatCompletionRequest::builder()
            .model("meta/llama-3.1-8b-instruct")
            .add_user_message("Hello!")
            .top_logprobs(5) // logprobs is not set to true
            .build();

        assert!(
            request.is_err(),
            "Request should fail when top_logprobs is set without logprobs being true"
        );
        Ok(())
    }

    #[ignore]
    #[test]
    fn test_chat_completions_max_tokens_out_of_range() -> Result<(), Box<dyn Error>> {
        let request = ChatCompletionRequest::builder()
            .model("meta/llama-3.1-8b-instruct")
            .add_user_message("Hello!")
            .max_tokens(4097) // assuming the model has a max context length of 4096
            .build();

        assert!(
            request.is_err(),
            "Request should fail when max_tokens exceeds model's context length"
        );
        Ok(())
    }

    #[test]
    fn test_chat_completions_invalid_top_p() -> Result<(), Box<dyn Error>> {
        let request = ChatCompletionRequest::builder()
            .model("meta/llama-3.1-8b-instruct")
            .add_user_message("Hello!")
            .top_p(1.5) // Invalid, should be between 0 and 1
            .build();

        assert!(
            request.is_err(),
            "Request should fail with invalid top_p value"
        );
        Ok(())
    }

    #[test]
    fn test_chat_completions_missing_messages() -> Result<(), Box<dyn Error>> {
        // Missing messages field in the request
        let request_result = ChatCompletionRequest::builder()
            .model("meta/llama-3.1-8b-instruct") // Valid model
            .build(); // This should fail because no messages are provided.

        assert!(
            request_result.is_err(),
            "Expected request to fail without messages."
        );

        if let Err(e) = request_result {
            println!("Expected error: {}", e); // Optionally print the error for debugging
        }

        Ok(())
    }

    #[test]
    fn test_chat_completions_negative_max_tokens() -> Result<(), Box<dyn Error>> {
        let request = ChatCompletionRequest::builder()
            .model("meta/llama-3.1-8b-instruct")
            .add_user_message("Hello, world!")
            .max_tokens(-10)
            .build();

        assert!(
            request.is_err(),
            "Request should fail with negative max_tokens"
        );

        Ok(())
    }

    #[ignore]
    #[test]
    fn test_chat_completions_unsupported_logit_bias() -> Result<(), Box<dyn Error>> {
        let request = ChatCompletionRequest::builder()
            .model("meta/llama-3.1-8b-instruct")
            .add_user_message("Hello, world!")
            .add_logit_bias("50256", -100)
            .build();

        assert!(request.is_err(), "Request should fail with logit_bias");

        Ok(())
    }

    #[test]
    fn test_chat_completions_invalid_temperature() -> Result<(), Box<dyn Error>> {
        let request = ChatCompletionRequest::builder()
            .model("meta/llama-3.1-8b-instruct")
            .add_user_message("Hello!")
            .temperature(2.5) // Invalid, should be between 0 and 2
            .build();

        assert!(
            request.is_err(),
            "Request should fail with invalid temperature"
        );

        Ok(())
    }

    #[test]
    fn test_chat_completions_max_stop_sequences() -> Result<(), Box<dyn Error>> {
        let request = ChatCompletionRequest::builder()
            .model("meta/llama-3.1-8b-instruct")
            .add_user_message("Tell me a story.")
            .stop(vec![
                "The end.".to_string(),
                "Once upon a time,".to_string(),
                "And then,".to_string(),
                "They lived happily ever after.".to_string(),
            ]) // 4 stop sequences, valid
            .build();

        assert!(
            request.is_ok(),
            "Request should succeed with 4 stop sequences"
        );
        Ok(())
    }

    #[test]
    fn test_chat_completions_large_stop_sequences() -> Result<(), Box<dyn Error>> {
        let request = ChatCompletionRequest::builder()
            .model("meta/llama-3.1-8b-instruct")
            .add_user_message("Tell me a story.")
            .stop(vec![
                "The end.".to_string(),
                "And so,".to_string(),
                "Once upon a time,".to_string(),
                "They lived happily ever after.".to_string(),
                "Unexpected stop.".to_string(),
            ])
            .build();

        assert!(
            request.is_err(),
            "Request should fail with too many stop sequences"
        );

        Ok(())
    }

    #[ignore]
    #[test]
    fn test_chat_completions_invalid_stop_sequences() -> Result<(), Box<dyn Error>> {
        let request = ChatCompletionRequest::builder()
            .model("meta/llama-3.1-8b-instruct")
            .add_user_message("Tell me a joke.")
            .stop(vec!["".to_string()])
            .build();

        assert!(
            request.is_err(),
            "Request should fail with invalid stop sequences"
        );

        Ok(())
    }

    #[ignore]
    #[test]
    fn test_chat_completions_presence_penalty_out_of_range() -> Result<(), Box<dyn Error>> {
        let request = ChatCompletionRequest::builder()
            .model("meta/llama-3.1-8b-instruct")
            .add_user_message("What's up?")
            .presence_penalty(3.0) // Out of valid range (-2.0 to 2.0)
            .build();

        assert!(
            request.is_err(),
            "Request should fail with invalid presence_penalty"
        );

        Ok(())
    }

    #[test]
    fn test_chat_completions_invalid_presence_penalty() -> Result<(), Box<dyn Error>> {
        let request = ChatCompletionRequest::builder()
            .model("meta/llama-3.1-8b-instruct")
            .add_user_message("What's up?")
            .presence_penalty(-2.5) // Invalid, should be between -2.0 and 2.0
            .build();

        assert!(
            request.is_err(),
            "Request should fail with invalid presence_penalty"
        );
        Ok(())
    }

    #[ignore]
    #[tokio::test]
    async fn test_chat_completions_with_user_field() -> Result<(), Box<dyn Error>> {
        let request = ChatCompletionRequest::builder()
            .model("meta/llama-3.1-8b-instruct")
            .add_user_message("Hi there!")
            .user("test_user")
            .build()
            .unwrap();

        // assert!(request.is_err(), "Request should fail with 'user' field");

        let result: Result<common::CompletionRequest> = request.try_into();

        assert!(
            result.is_err(),
            "Conversion should fail with 'user' field set",
        );

        Ok(())
    }

    #[test]
    fn test_chat_completions_valid_with_seed() -> Result<(), Box<dyn Error>> {
        let request = ChatCompletionRequest::builder()
            .model("meta/llama-3.1-8b-instruct")
            .add_user_message("Repeatable result")
            .seed(12345)
            .build();

        assert!(
            request.is_ok(),
            "Request should succeed with seed value for determinism"
        );
        Ok(())
    }

    #[test]
    fn test_validate_chat_messages_multiple_system_messages() -> Result<(), Box<dyn Error>> {
        let request = ChatCompletionRequest::builder()
            .model("test-model")
            .add_system_message("System message 1")
            .add_system_message("System message 2")
            .add_user_message("Hello!")
            .build()?;

        let result = validate_and_collect_chat_messages(request.messages.clone());
        assert!(result.is_err());
        if let Err(e) = result {
            assert_eq!(e.to_string(), "More than one system message found");
        }

        Ok(())
    }

    #[test]
    fn test_validate_chat_messages_user_messages_do_not_alternate() -> Result<(), Box<dyn Error>> {
        let request = ChatCompletionRequest::builder()
            .model("test-model")
            .add_user_message("Hello!")
            .add_user_message("How are you?")
            .build()?;

        let result = validate_and_collect_chat_messages(request.messages.clone());
        assert!(result.is_err());

        if let Err(e) = result {
            assert_eq!(e.to_string(), "User and assistant messages must alternate");
        }

        Ok(())
    }

    #[ignore]
    #[test]
    fn test_validate_chat_messages_user_message_not_text() -> Result<(), Box<dyn Error>> {
        let message = ChatCompletionMessage {
            role: MessageRole::user,
            content: Content::ImageUrl(vec![ImageUrl {
                r#type: ContentType::image_url,
                text: None,
                image_url: Some(ImageUrlType {
                    url: "http://example.com/image.png".to_string(),
                }),
            }]),
            name: None,
        };

        let request = ChatCompletionRequest::builder()
            .model("test-model")
            .add_message(message)
            .build()?;

        let result = validate_and_collect_chat_messages(request.messages.clone());
        assert!(result.is_err());

        if let Err(e) = result {
            assert_eq!(e.to_string(), "Generic error: User message must be text");
        }

        Ok(())
    }

    #[test]
    fn test_try_from_chat_completion_request_with_unsupported_fields() -> Result<(), Box<dyn Error>>
    {
        let request = ChatCompletionRequest::builder()
            .model("test-model")
            .add_user_message("Hello!")
            .response_format(Some(json!({"format": "unsupported"})))
            .tools(Some(vec![Tool {
                r#type: ToolType::Function,
                function: Function {
                    name: "test_function".to_string(),
                    description: None,
                    parameters: FunctionParameters {
                        schema_type: JSONSchemaType::Object,
                        properties: None,
                        required: None,
                    },
                },
            }]))
            .tool_choice(Some(ToolChoiceType::Auto))
            .build()?;

        let result: Result<common::CompletionRequest> = request.try_into();
        assert!(
            result.is_err(),
            "Conversion should fail with unsupported fields"
        );

        Ok(())
    }

    #[test]
    fn test_deserialize_content_with_image_urls() {
        let json_data = r#"
    {
        "role": "assistant",
        "content": [
            "This is a text message.",
            "https://example.com/image1.png",
            "Another text message.",
            "https://example.com/image2.png"
        ]
    }
    "#;

        let message: ChatCompletionMessage =
            serde_json::from_str(json_data).expect("Deserialization failed");

        if let Content::ImageUrl(parts) = message.content {
            assert_eq!(parts.len(), 4);
            assert_eq!(parts[0].r#type, ContentType::text);
            assert_eq!(parts[0].text.as_ref().unwrap(), "This is a text message.");
            assert_eq!(parts[1].r#type, ContentType::image_url);
            assert_eq!(
                parts[1].image_url.as_ref().unwrap().url,
                "https://example.com/image1.png"
            );
        } else {
            panic!("Expected Content::ImageUrl");
        }
    }

    #[test]
    fn test_try_from_chat_completion_request_success() -> Result<(), Box<dyn Error>> {
        let request = ChatCompletionRequest::builder()
            .model("test-model")
            .add_user_message("Hello!")
            .add_assistant_message("Hi there!")
            .add_user_message("How are you?")
            .build()?;

        let completion_request: common::CompletionRequest = request.try_into()?;

        assert!(matches!(
            completion_request.prompt,
            common::PromptType::ChatCompletion(_)
        ));

        Ok(())
    }

    #[test]
    fn test_chat_completion_sampling_params_with_valid_nvext() {
        let nvext = NvExt {
            ignore_eos: Some(true),
            repetition_penalty: Some(0.6),
            top_k: Some(3),
            use_raw_prompt: None,
            greed_sampling: None,
            annotations: None,
        };
        let request = ChatCompletionRequest::builder()
            .nvext(nvext)
            .model("foo")
            .add_system_message("Hello!")
            .build()
            .expect("Failed to build request with valid nvext");

        assert_eq!(request.nvext.as_ref().unwrap().ignore_eos, Some(true));
        assert_eq!(
            request.nvext.as_ref().unwrap().repetition_penalty,
            Some(0.6)
        );
        assert_eq!(request.nvext.as_ref().unwrap().top_k, Some(3));
    }

    #[test]
    fn test_completion_sampling_params_without_nvext() {
        let request = ChatCompletionRequest::builder()
            .model("foo")
            .add_user_message("Test")
            .build()
            .unwrap();

        assert_eq!(request.frequency_penalty, None);
        assert_eq!(request.logprobs, None);
    }

    #[test]
    fn test_completion_sampling_params_with_valid_nvext() {
        let nvext = NvExt {
            ignore_eos: Some(true),
            repetition_penalty: Some(0.6),
            top_k: Some(3),
            ..Default::default()
        };
        let request = ChatCompletionRequest::builder()
            .nvext(nvext)
            .model("foo")
            .add_user_message("Test")
            .build()
            .expect("Failed to build request with valid nvext");

        assert_eq!(request.nvext.as_ref().unwrap().ignore_eos, Some(true));
        assert_eq!(
            request.nvext.as_ref().unwrap().repetition_penalty,
            Some(0.6)
        );
        assert_eq!(request.nvext.as_ref().unwrap().top_k, Some(3));
    }

    // #[test]
    // fn test_normalize_unicode_characters() {
    //     let str = "Hello there how are you\u{E0020}?".to_string();
    //     let normalized = str.sanitize_text();

    //     assert_eq!(normalized, "Hello there how are you?");
    // }

    // #[tokio::test]
    // async fn test_chat_completion_request_filtered() {
    //     // Define input messages with Unicode character to filter
    //     let messages = vec![
    //         ChatCompletionMessage {
    //             role: MessageRole::user,
    //             content: Content::Text(
    //                 "Hello there how are you\u{E0020}?"
    //                     .to_string()
    //                     .normalize_unicode_characters(),
    //             ),
    //             name: None,
    //         },
    //         ChatCompletionMessage {
    //             role: MessageRole::assistant,
    //             content: Content::Text("How may I help you?".to_string()),
    //             name: None,
    //         },
    //         ChatCompletionMessage {
    //             role: MessageRole::user,
    //             content: Content::Text("Do something for me?".to_string()),
    //             name: None,
    //         },
    //     ];

    //     // Define expected filtered messages
    //     let expected = vec![
    //         ChatCompletionMessage {
    //             role: MessageRole::user,
    //             content: Content::Text("Hello there how are you?".to_string()),
    //             name: None,
    //         },
    //         ChatCompletionMessage {
    //             role: MessageRole::assistant,
    //             content: Content::Text("How may I help you?".to_string()),
    //             name: None,
    //         },
    //         ChatCompletionMessage {
    //             role: MessageRole::user,
    //             content: Content::Text("Do something for me?".to_string()),
    //             name: None,
    //         },
    //     ];

    //     // Build ChatCompletionRequest with filtering applied
    //     let request = ChatCompletionRequest::builder()
    //         .model("foo")
    //         .messages(messages)
    //         .build()
    //         .expect("Failed to build ChatCompletionRequest");

    //     // Validate each message matches the expected filtered content
    //     for (i, message) in request.messages.iter().enumerate() {
    //         assert_eq!(message.role, expected[i].role);
    //         if let Content::Text(ref content) = message.content {
    //             if let Content::Text(ref expected_content) = expected[i].content {
    //                 assert_eq!(content, expected_content);
    //             }
    //         }
    //     }
    // }
}
