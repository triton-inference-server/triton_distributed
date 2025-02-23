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

use std::collections::HashMap;

use derive_builder::Builder;
use serde::{Deserialize, Serialize};
use validator::Validate;

mod aggregator;
mod delta;

pub use aggregator::DeltaAggregator;

use super::{
    common::{self, SamplingOptionsProvider, StopConditionsProvider},
    nvext::{NvExt, NvExtProvider},
    validate_logit_bias, CompletionUsage, ContentProvider, OpenAISamplingOptionsProvider,
    OpenAIStopConditionsProvider, MAX_FREQUENCY_PENALTY, MAX_PRESENCE_PENALTY, MAX_TEMPERATURE,
    MAX_TOP_P, MIN_FREQUENCY_PENALTY, MIN_PRESENCE_PENALTY, MIN_TEMPERATURE, MIN_TOP_P,
};

use triton_distributed::protocols::annotated::AnnotationsProvider;

/// Legacy OpenAI CompletionRequest
///
/// Reference: <https://platform.openai.com/docs/api-reference/completions>
#[derive(Serialize, Deserialize, Builder, Validate, Debug, Clone)]
#[builder(build_fn(private, name = "build_internal", validate = "Self::validate"))]
pub struct CompletionRequest {
    /// ID of the model to use.
    #[builder(setter(into))]
    pub model: String,

    /// The prompt(s) to generate completions for, encoded as a string, array of
    /// strings, array of tokens, or array of token arrays.
    ///
    /// NIM Compatibility:
    /// The NIM LLM API only supports a single prompt as a string at this time.
    #[builder(setter(into))]
    pub prompt: String,

    /// The maximum number of tokens that can be generated in the completion.
    /// The token count of your prompt plus max_tokens cannot exceed the model's context length.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into, strip_option))]
    pub max_tokens: Option<i32>,

    /// The minimum number of tokens to generate. We ignore stop tokens until we see this many
    /// tokens. Leave this None unless you are working on the pre-processor.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into, strip_option))]
    pub min_tokens: Option<i32>,

    /// If set, partial message deltas will be sent, like in ChatGPT. Tokens will be sent as data-only
    /// server-sent events as they become available, with the stream terminated by a data: \[DONE\]
    ///
    /// If this is set to true, but the response cannot be streamed an error will be returned.
    ///
    /// NIM Compatibility:
    /// The NIM SDK can send extra meta data in the SSE stream using the `:` comment, `event:`,
    /// or `id:` fields. See the `enable_sse_metadata` field in the NvExt object.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub stream: Option<bool>,

    /// How many completions to generate for each prompt.
    ///
    /// Note: Because this parameter generates many completions, it can quickly consume your token quota.
    /// Use carefully and ensure that you have reasonable settings for `max_tokens` and `stop`.
    ///
    /// NIM Compatibility:
    /// At this time, the NIM LLM API does not support `n` completions.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into, strip_option))]
    pub n: Option<i32>,

    /// Generates `best_of` completions server-side and returns the "best" (the one with the
    /// highest log probability per token). Results cannot be streamed.
    ///
    /// When used with `n`, best_of controls the number of candidate completions and `n` specifies
    /// how many to return – `best_of` must be greater than `n`.
    ///
    /// NIM Compatibility:
    /// At this time, the NIM LLM API does not support `best_of` completions.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into, strip_option))]
    pub best_of: Option<i32>,

    /// What sampling `temperature` to use, between 0 and 2. Higher values like 0.8 will make the
    /// output more random, while lower values like 0.2 will make it more focused and deterministic.
    ///
    /// We generally recommend altering this or `top_p` but not both.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(range(min = "MIN_TEMPERATURE", max = "MAX_TEMPERATURE"))]
    #[builder(default, setter(into, strip_option))]
    pub temperature: Option<f32>,

    /// An alternative to sampling with `temperature`, called nucleus sampling, where the model
    /// considers the results of the tokens with `top_p` probability mass. So 0.1 means only the tokens
    /// comprising the top 10% probability mass are considered.
    ///
    /// We generally recommend altering this or `temperature` but not both.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(range(min = "MIN_TOP_P", max = "MAX_TOP_P"))]
    #[builder(default, setter(into, strip_option))]
    pub top_p: Option<f32>,

    /// Include the log probabilities on the logprobs most likely output tokens, as well the chosen tokens.
    /// For example, if logprobs is 5, the API will return a list of the 5 most likely tokens. The API will
    /// always return the logprob of the sampled token, so there may be up to logprobs+1 elements in the
    /// response.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into, strip_option))]
    pub logprobs: Option<i32>,

    /// Echo back the prompt in addition to the completion
    ///
    /// NIM Compatibility:
    /// At this time, the NIM LLM API does not support `echo` completions.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub echo: Option<bool>,

    /// Up to 4 sequences where the API will stop generating further tokens. The returned text will not
    /// contain the stop sequence.
    #[serde(skip_serializing_if = "Option::is_none")]
    // #[builder(default, setter(into, strip_option))]
    #[builder(default, setter(strip_option))]
    pub stop: Option<Vec<String>>,

    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency
    /// in the text so far, decreasing the model's likelihood to repeat the same line verbatim.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(range(min = "MIN_FREQUENCY_PENALTY", max = "MAX_FREQUENCY_PENALTY"))]
    #[builder(default, setter(into, strip_option))]
    pub frequency_penalty: Option<f32>,

    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in
    /// the text so far, increasing the model's likelihood to talk about new topics.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(range(min = "MIN_PRESENCE_PENALTY", max = "MAX_PRESENCE_PENALTY"))]
    #[builder(default, setter(into, strip_option))]
    pub presence_penalty: Option<f32>,

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
    /// ```rust
    /// use triton_llm::protocols::openai::completions::CompletionRequest;
    ///
    /// let request = CompletionRequest::builder()
    ///     .prompt("What is the meaning of life?")
    ///     .model("gpt-3.5-turbo")
    ///     .add_logit_bias(1337, -100) // using an int as a key is ok
    ///     .add_logit_bias("42", 100)  // using a string as a key is also ok
    ///     .build()
    ///     .expect("Should not fail");
    ///
    /// assert!(CompletionRequest::builder()
    ///     .prompt("What is the meaning of life?")
    ///     .model("gpt-3.5-turbo")
    ///     .add_logit_bias("some non int", -100)
    ///     .build()
    ///     .is_err());
    /// ```
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(custom(function = "validate_logit_bias"))]
    #[builder(default)]
    pub logit_bias: Option<HashMap<String, i32>>,

    /// A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse.
    ///
    /// NIM Compatibility:
    /// If provided, then the value of this field will be included in the trace metadata and the accounting
    /// data (if enabled).
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into, strip_option))]
    pub user: Option<String>,

    /// OpenAI specific API parameter; this is not supported by NIM models; however,
    /// is preserved as part of the API for compatibility.
    ///
    /// OpenAI API Reference:
    /// <https://platform.openai.com/docs/api-reference/completions/create>
    ///
    /// A validation error will be thrown if this field is set when executing against
    /// any NIM model.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into, strip_option))]
    pub suffix: Option<String>,

    /// NVIDIA extension to OpenAI's legacy v1::completion::CompletionRequest
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub nvext: Option<NvExt>,
}

impl CompletionRequest {
    /// Create a new CompletionRequestBuilder
    pub fn builder() -> CompletionRequestBuilder {
        CompletionRequestBuilder::default()
    }
}

impl CompletionRequestBuilder {
    // This is a pre-build validate function
    // This is called before the generated build method, in this case build_internal, is called
    // This has access to the internal state of the builder
    fn validate(&self) -> Result<(), String> {
        Ok(())
    }

    /// Builds and validates the CompletionRequest
    ///
    /// ```rust
    /// use triton_llm::protocols::openai::completions::CompletionRequest;
    ///
    /// let request = CompletionRequest::builder()
    ///     .model("mixtral-8x7b-instruct-v0.1")
    ///     .prompt("Hello")
    ///     .max_tokens(16)
    ///     .build()
    ///     .expect("Failed to build CompletionRequest");
    /// ```
    pub fn build(&self) -> anyhow::Result<CompletionRequest> {
        // Calls the build_internal, validates the result, then performs addition
        // post build validation. This is where we might handle any mutually exclusive fields
        // and ensure there are no collisions.
        let request = self
            .build_internal()
            .map_err(|e| anyhow::anyhow!("Failed to build CompletionRequest: {}", e))?;

        request
            .validate()
            .map_err(|e| anyhow::anyhow!("Failed to validate CompletionRequest: {}", e))?;

        Ok(request)
    }

    /// Add a stop condition to the `Vec<String>` in the ChatCompletionRequest
    /// This will either create or append to the `Vec<String>`
    pub fn add_stop(&mut self, stop: impl Into<String>) -> &mut Self {
        if self.stop.is_none() {
            self.stop = Some(Some(vec![]));
        }
        self.stop
            .as_mut()
            .unwrap()
            .as_mut()
            .unwrap()
            .push(stop.into());
        self
    }

    /// Add a tool to the `HashMap<String, i32>` in the ChatCompletionRequest
    /// This will either create or update the `HashMap<String, i32>`
    pub fn add_logit_bias<T>(&mut self, key: T, value: i32) -> &mut Self
    where
        T: std::fmt::Display,
    {
        if self.logit_bias.is_none() {
            self.logit_bias = Some(Some(HashMap::new()));
        }
        self.logit_bias
            .as_mut()
            .unwrap()
            .as_mut()
            .unwrap()
            .insert(key.to_string(), value);
        self
    }
}

/// Legacy OpenAI CompletionResponse
/// Represents a completion response from the API.
/// Note: both the streamed and non-streamed response objects share the same
/// shape (unlike the chat endpoint).
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CompletionResponse {
    /// A unique identifier for the completion.
    pub id: String,

    /// The list of completion choices the model generated for the input prompt.
    pub choices: Vec<CompletionChoice>,

    /// The Unix timestamp (in seconds) of when the completion was created.
    pub created: u64,

    /// The model used for completion.
    pub model: String,

    /// The object type, which is always "text_completion"
    pub object: String,

    /// Usage statistics for the completion request.
    pub usage: Option<CompletionUsage>,

    /// This fingerprint represents the backend configuration that the model runs with.
    /// Can be used in conjunction with the seed request parameter to understand when backend
    /// changes have been made that might impact determinism.
    ///
    /// NIM Compatibility:
    /// This field is not supported by the NIM; however it will be added in the future.
    /// The optional nature of this field will be relaxed when it is supported.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
    // TODO(ryan)
    // pub nvext: Option<NimResponseExt>,
}

/// Legacy OpenAI CompletionResponse Choice component
#[derive(Clone, Debug, Deserialize, Serialize, Builder)]
pub struct CompletionChoice {
    #[builder(setter(into))]
    pub text: String,

    #[builder(default = "0")]
    pub index: u64,

    #[builder(default, setter(into, strip_option))]
    pub finish_reason: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub logprobs: Option<LogprobResult>,
}

impl ContentProvider for CompletionChoice {
    fn content(&self) -> String {
        self.text.clone()
    }
}

impl CompletionChoice {
    pub fn builder() -> CompletionChoiceBuilder {
        CompletionChoiceBuilder::default()
    }
}

// TODO: validate this is the correct format
/// Legacy OpenAI LogprobResult component
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct LogprobResult {
    pub tokens: Vec<String>,
    pub token_logprobs: Vec<f32>,
    pub top_logprobs: Vec<HashMap<String, f32>>,
    pub text_offset: Vec<i32>,
}

impl NvExtProvider for CompletionRequest {
    fn nvext(&self) -> Option<&NvExt> {
        self.nvext.as_ref()
    }

    fn raw_prompt(&self) -> Option<String> {
        if let Some(nvext) = self.nvext.as_ref() {
            if let Some(use_raw_prompt) = nvext.use_raw_prompt {
                if use_raw_prompt {
                    return Some(self.prompt.clone());
                }
            }
        }
        None
    }
}

impl AnnotationsProvider for CompletionRequest {
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

impl OpenAISamplingOptionsProvider for CompletionRequest {
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

impl OpenAIStopConditionsProvider for CompletionRequest {
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

#[derive(Builder)]
pub struct ResponseFactory {
    #[builder(setter(into))]
    pub model: String,

    #[builder(default)]
    pub system_fingerprint: Option<String>,

    #[builder(default = "format!(\"cmpl-{}\", uuid::Uuid::new_v4())")]
    pub id: String,

    #[builder(default = "\"text_completion\".to_string()")]
    pub object: String,

    #[builder(default = "chrono::Utc::now().timestamp() as u64")]
    pub created: u64,
}

impl ResponseFactory {
    pub fn builder() -> ResponseFactoryBuilder {
        ResponseFactoryBuilder::default()
    }

    pub fn make_response(
        &self,
        choice: CompletionChoice,
        usage: Option<CompletionUsage>,
    ) -> CompletionResponse {
        CompletionResponse {
            id: self.id.clone(),
            object: self.object.clone(),
            created: self.created,
            model: self.model.clone(),
            choices: vec![choice],
            system_fingerprint: self.system_fingerprint.clone(),
            usage,
        }
    }
}

/// Implements TryFrom for converting an OpenAI's CompletionRequest to an Engine's CompletionRequest
impl TryFrom<CompletionRequest> for common::CompletionRequest {
    type Error = anyhow::Error;

    fn try_from(request: CompletionRequest) -> Result<Self, Self::Error> {
        // openai_api_rs::v1::completion::CompletionRequest {
        // NA  pub model: String,
        //     pub prompt: String,
        // **  pub suffix: Option<String>,
        //     pub max_tokens: Option<i32>,
        //     pub temperature: Option<f32>,
        //     pub top_p: Option<f32>,
        //     pub n: Option<i32>,
        //     pub stream: Option<bool>,
        //     pub logprobs: Option<i32>,
        //     pub echo: Option<bool>,
        //     pub stop: Option<Vec<String, Global>>,
        //     pub presence_penalty: Option<f32>,
        //     pub frequency_penalty: Option<f32>,
        //     pub best_of: Option<i32>,
        //     pub logit_bias: Option<HashMap<String, i32, RandomState>>,
        //     pub user: Option<String>,
        // }
        //
        // ** no supported

        if request.suffix.is_some() {
            return Err(anyhow::anyhow!("suffix is not supported"));
        }

        let stop_conditions = request
            .extract_stop_conditions()
            .map_err(|e| anyhow::anyhow!("Failed to extract stop conditions: {}", e))?;

        let sampling_options = request
            .extract_sampling_options()
            .map_err(|e| anyhow::anyhow!("Failed to extract sampling options: {}", e))?;

        let prompt = common::PromptType::Completion(common::CompletionContext {
            prompt: request.prompt,
            system_prompt: None,
        });

        Ok(common::CompletionRequest {
            prompt,
            stop_conditions,
            sampling_options,
            mdc_sum: None,
            annotations: None,
        })
    }
}

impl TryFrom<common::StreamingCompletionResponse> for CompletionChoice {
    type Error = anyhow::Error;

    fn try_from(response: common::StreamingCompletionResponse) -> Result<Self, Self::Error> {
        let choice = CompletionChoice {
            text: response
                .delta
                .text
                .ok_or(anyhow::anyhow!("No text in response"))?,
            index: response.delta.index.unwrap_or(0) as u64,
            logprobs: None,
            finish_reason: match &response.delta.finish_reason {
                Some(common::FinishReason::EoS) => Some("stop".to_string()),
                Some(common::FinishReason::Stop) => Some("stop".to_string()),
                Some(common::FinishReason::Length) => Some("length".to_string()),
                Some(common::FinishReason::Error(err_msg)) => {
                    return Err(anyhow::anyhow!("finish_reason::error = {}", err_msg));
                }
                Some(common::FinishReason::Cancelled) => Some("cancelled".to_string()),
                None => None,
            },
        };

        Ok(choice)
    }
}
