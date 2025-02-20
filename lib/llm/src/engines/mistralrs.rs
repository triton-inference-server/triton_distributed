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

use std::{cmp::min, num::NonZero, path::Path, sync::Arc};

use async_stream::stream;
use async_trait::async_trait;
use either::Either;
use indexmap::IndexMap;
use mistralrs::{
    Constraint, DefaultSchedulerMethod, Device, DeviceMapMetadata, DeviceMapSetting,
    GGUFLoaderBuilder, GGUFSpecificConfig, MemoryGpuConfig, MistralRs, MistralRsBuilder,
    ModelDType, NormalRequest, PagedAttentionConfig, Pipeline, Request, RequestMessage, ResponseOk,
    SamplingParams, SchedulerConfig, TokenSource,
};
use tokio::sync::mpsc::channel;

use triton_distributed_runtime::engine::{AsyncEngine, AsyncEngineContextProvider, ResponseStream};
use triton_distributed_runtime::pipeline::error as pipeline_error;
use triton_distributed_runtime::pipeline::{Error, ManyOut, SingleIn};
use triton_distributed_runtime::protocols::annotated::Annotated;

use crate::protocols::openai::chat_completions::{
    ChatCompletionChoiceDelta, ChatCompletionContent, ChatCompletionRequest,
    ChatCompletionResponseDelta, Content, FinishReason, MessageRole,
};
use crate::types::openai::chat_completions::OpenAIChatCompletionsStreamingEngine;

/// If user does not provide a max_tokens limit prompt+output to this many
const DEFAULT_MAX_TOKENS: i32 = 8192;

pub async fn make_engine(
    gguf_path: &Path,
) -> pipeline_error::Result<OpenAIChatCompletionsStreamingEngine> {
    let engine = MistralRsEngine::new(gguf_path).await?;
    let engine: OpenAIChatCompletionsStreamingEngine = Arc::new(engine);
    Ok(engine)
}

/// Gets the best device, cpu, cuda if compiled with CUDA
fn best_device() -> pipeline_error::Result<Device> {
    #[cfg(not(feature = "metal"))]
    {
        Ok(Device::cuda_if_available(0)?)
    }
    #[cfg(feature = "metal")]
    {
        Ok(Device::new_metal(0)?)
    }
}

struct MistralRsEngine {
    mistralrs: Arc<MistralRs>,
    pipeline: Arc<tokio::sync::Mutex<dyn Pipeline + Send + Sync + 'static>>,
}

impl MistralRsEngine {
    async fn new(model_path: &Path) -> pipeline_error::Result<Self> {
        let Some(model_filename) = model_path.file_name() else {
            pipeline_error::bail!("Missing filename in model path");
        };
        let Some(model_dir) = model_path.parent() else {
            pipeline_error::bail!("Invalid model path");
        };

        // Select a Mistral model
        // We do not use any files from HF servers here, and instead load the
        // chat template from the specified file, and the tokenizer and model from a
        // local GGUF file at the path `.`
        let loader = GGUFLoaderBuilder::new(
            None,
            None,
            model_dir.display().to_string(),
            vec![model_filename.to_string_lossy().into_owned()],
            GGUFSpecificConfig {
                prompt_chunksize: None,
                topology: None,
            },
        )
        .build();

        // Paged attention requires cuda
        let paged_attention_config = if cfg!(feature = "cuda") {
            Some(PagedAttentionConfig::new(
                Some(32),
                1024,
                MemoryGpuConfig::Utilization(0.9),
            )?)
        } else {
            None
        };
        // Load, into a Pipeline
        let pipeline = loader.load_model_from_hf(
            None,
            TokenSource::CacheToken,
            &ModelDType::Auto,
            &best_device()?,
            false,
            DeviceMapSetting::Map(DeviceMapMetadata::dummy()),
            None,
            paged_attention_config,
        )?;
        let scheduler = if cfg!(feature = "cuda") {
            tracing::debug!("Using mistralrs PagedAttentionMeta scheduler");
            let config = match pipeline.lock().await.get_metadata().cache_config.as_ref() {
                Some(conf) => conf.clone(),
                None => {
                    anyhow::bail!("Failed loading model config");
                }
            };
            SchedulerConfig::PagedAttentionMeta {
                max_num_seqs: 5,
                config,
            }
        } else {
            tracing::debug!("Using mistralrs DefaultScheduler");
            SchedulerConfig::DefaultScheduler {
                // Safety: unwrap trivially safe here
                method: DefaultSchedulerMethod::Fixed(NonZero::new(5).unwrap()),
            }
        };
        // Create the MistralRs, which is a runner
        let builder = MistralRsBuilder::new(pipeline.clone(), scheduler);
        Ok(MistralRsEngine {
            mistralrs: builder.build(),
            pipeline,
        })
    }
}

#[async_trait]
impl
    AsyncEngine<
        SingleIn<ChatCompletionRequest>,
        ManyOut<Annotated<ChatCompletionResponseDelta>>,
        Error,
    > for MistralRsEngine
{
    async fn generate(
        &self,
        request: SingleIn<ChatCompletionRequest>,
    ) -> Result<ManyOut<Annotated<ChatCompletionResponseDelta>>, Error> {
        let (request, context) = request.transfer(());
        let ctx = context.context();
        let (tx, mut rx) = channel(10_000);
        let maybe_tok = self.pipeline.lock().await.tokenizer();

        let mut prompt_tokens = 0;
        let mut messages = vec![];
        for m in request.messages {
            let content = match m.content {
                Content::Text(prompt) => {
                    if let Some(tok) = maybe_tok.as_ref() {
                        prompt_tokens = tok
                            .encode(prompt.clone(), false)
                            .map(|e| e.len() as i32)
                            .unwrap_or(0);
                    }
                    prompt
                }
                Content::ImageUrl(_) => {
                    anyhow::bail!("Content::ImageUrl type is not supported");
                }
            };
            let r = IndexMap::from([
                ("role".to_string(), Either::Left(m.role.to_string())),
                ("content".to_string(), Either::Left(content)),
            ]);
            messages.push(r);
        }
        if messages.is_empty() {
            anyhow::bail!("Empty request");
        }
        // TODO tracing::trace print the latest prompt, which should be the last message at user
        // level.
        //tracing::info!(prompt_tokens, "Received prompt");
        let limit = DEFAULT_MAX_TOKENS - prompt_tokens;
        let max_output_tokens = min(request.max_tokens.unwrap_or(limit), limit);

        let mistralrs_request = Request::Normal(NormalRequest {
            messages: RequestMessage::Chat(messages),
            sampling_params: SamplingParams::deterministic(),
            response: tx,
            return_logprobs: false,
            is_streaming: true,
            id: self.mistralrs.next_request_id(),
            constraint: Constraint::None,
            suffix: None,
            adapters: None,
            tools: None,
            tool_choice: None,
            logits_processors: None,
            return_raw_logits: false,
        });

        self.mistralrs.get_sender()?.send(mistralrs_request).await?;

        let mut used_output_tokens = 0;
        let output = stream! {
            while let Some(response) = rx.recv().await {
                let response = match response.as_result() {
                    Ok(r) => r,
                    Err(err) => {
                        tracing::error!(%err, "Failed converting mistralrs channel response to result.");
                        break;
                    }
                };
                match response {
                    ResponseOk::Chunk(c) => {
                        let Some(from_assistant) = c.choices[0].delta.content.clone() else {
                            tracing::warn!("No content from mistralrs. Abandoning request.");
                            break;
                        };
                        if let Some(tok) = maybe_tok.as_ref() {
                            used_output_tokens += tok
                                .encode(from_assistant.clone(), false)
                                .map(|e| e.len() as i32)
                                .unwrap_or(0);
                        }
                        let finish_reason = match &c.choices[0].finish_reason {
                            Some(fr) => Some(fr.parse::<FinishReason>().unwrap_or(FinishReason::null)),
                            None if used_output_tokens >= max_output_tokens => {
                                tracing::debug!(used_output_tokens, max_output_tokens, "Met or exceed max_tokens. Stopping.");
                                Some(FinishReason::length)
                            }
                            None => None,
                        };
                        //tracing::trace!("from_assistant: {from_assistant}");

                        let delta = ChatCompletionResponseDelta{
                            id: c.id,
                            choices: vec![ChatCompletionChoiceDelta{
                                index: 0,
                                delta: ChatCompletionContent{
                                    //role: c.choices[0].delta.role,
                                    role: Some(MessageRole::assistant),
                                    content: Some(from_assistant),
                                    tool_calls: None,
                                },
                                logprobs: None,
                                finish_reason,
                            }],
                            model: c.model,
                            created: c.created as u64,
                            object: c.object.clone(),
                            usage: None,
                            system_fingerprint: Some(c.system_fingerprint),
                            service_tier: None,
                        };
                        let ann = Annotated{
                            id: None,
                            data: Some(delta),
                            event: None,
                            comment: None,
                        };
                        yield ann;

                        if finish_reason.is_some() {
                            //tracing::trace!("Finish reason: {finish_reason:?}");
                            break;
                        }
                    },
                    x => tracing::error!("Unhandled. {x:?}"),
                }
            }
        };
        Ok(ResponseStream::new(Box::pin(output), ctx))
    }
}
