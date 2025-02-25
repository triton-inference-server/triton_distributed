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

use serde::{Deserialize, Serialize};
use triton_distributed_llm::protocols::{
    common,
    openai::{
        self,
        completions::{CompletionRequest, CompletionRequestBuilder},
        nvext::NvExt,
    },
};

#[derive(Serialize, Deserialize, Debug, Clone)]
struct CompletionSample {
    request: CompletionRequest,
    description: String,
}

impl CompletionSample {
    fn new<F>(description: impl Into<String>, configure: F) -> Result<Self, String>
    where
        F: FnOnce(&mut CompletionRequestBuilder) -> &mut CompletionRequestBuilder,
    {
        let mut builder = CompletionRequestBuilder::default();
        builder
            .model("gpt-3.5-turbo")
            .prompt("What is the meaning of life?");
        configure(&mut builder);
        Ok(Self {
            request: builder.build().unwrap(),
            description: description.into(),
        })
    }
}

#[test]
fn minimum_viable_request() {
    let request = CompletionRequest::builder()
        .prompt("What is the meaning of life?")
        .model("gpt-3.5-turbo")
        .build()
        .expect("error building request");

    insta::assert_json_snapshot!(request);
}

#[test]
fn missing_model() {
    let request = CompletionRequest::builder()
        .prompt("What is the meaning of life?")
        .build();
    assert!(request.is_err());
}

#[test]
fn missing_prompt() {
    let request = CompletionRequest::builder().model("gpt-3.5-turbo").build();
    assert!(request.is_err());
}

#[test]
fn out_of_range() {
    let request = CompletionRequest::builder()
        .prompt("What is the meaning of life?")
        .model("gpt-3.5-turbo")
        .temperature(openai::MAX_TEMPERATURE + 1.0)
        .build();
    assert!(request.is_err());

    let request = CompletionRequest::builder()
        .prompt("What is the meaning of life?")
        .model("gpt-3.5-turbo")
        .temperature(openai::MIN_TEMPERATURE - 1.0)
        .build();
    assert!(request.is_err());
}

#[test]
fn ignore_eos() {
    let request = CompletionRequest::builder()
        .prompt("What is the meaning of life?")
        .model("gpt-3.5-turbo")
        .nvext(
            NvExt::builder()
                .ignore_eos(true)
                .build()
                .expect("error building nvext"),
        )
        .build()
        .expect("error building request");

    let request = common::CompletionRequest::try_from(request).expect("error converting request");

    let ignore_eos = request.stop_conditions.ignore_eos.unwrap();
    assert!(ignore_eos);
}

#[test]
fn valid_samples() {
    let mut settings = insta::Settings::clone_current();
    settings.set_sort_maps(true);
    let _guard = settings.bind_to_scope();

    let samples = build_samples().expect("error building samples");

    // iteration on all sample and call validate and expect it to be ok
    for sample in &samples {
        insta::with_settings!({
            description => &sample.description,
        }, {
        insta::assert_json_snapshot!(sample.request);
        });
    }
}
#[allow(clippy::vec_init_then_push)]
fn build_samples() -> Result<Vec<CompletionSample>, String> {
    let mut samples = Vec::new();

    samples.push(CompletionSample::new(
        "should have only prompt and model fields",
        |builder| builder,
    )?);

    samples.push(CompletionSample::new(
        "should have prompt, model, and max_tokens fields",
        |builder| builder.max_tokens(10),
    )?);

    samples.push(CompletionSample::new(
        "should have prompt, model, and temperature fields",
        |builder| builder.temperature(openai::MIN_TEMPERATURE),
    )?);

    samples.push(CompletionSample::new(
        "should have prompt, model, and top_p fields",
        |builder| builder.top_p(openai::MIN_TOP_P),
    )?);

    samples.push(CompletionSample::new(
        "should have prompt, model, and frequency_penalty fields",
        |builder| builder.frequency_penalty(openai::MIN_FREQUENCY_PENALTY),
    )?);

    samples.push(CompletionSample::new(
        "should have prompt, model, and presence_penalty fields",
        |builder| builder.presence_penalty(openai::MIN_PRESENCE_PENALTY),
    )?);

    samples.push(CompletionSample::new(
        "should have prompt, model, and stop fields",
        |builder| builder.stop(vec!["\n".to_string()]),
    )?);

    samples.push(CompletionSample::new(
        "should have prompt, model, and echo fields",
        |builder| builder.echo(true),
    )?);

    samples.push(CompletionSample::new(
        "should have prompt, model, and stream fields",
        |builder| builder.stream(true),
    )?);

    samples.push(CompletionSample::new(
        "should have prompt, model, and logit_bias fields with the logits_bias having two key/value pairs",
        |builder| builder.add_logit_bias(1337, -100).add_logit_bias("42", 100),
    )?);

    Ok(samples)
}
