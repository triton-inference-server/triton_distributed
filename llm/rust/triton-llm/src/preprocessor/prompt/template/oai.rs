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

use super::*;

use minijinja::{context, value::Value};

use crate::protocols::openai::{
    chat_completions::{ChatCompletionMessage, ChatCompletionRequest, Content, MessageRole},
    completions::CompletionRequest,
};
use tracing;

impl OAIChatLikeRequest for ChatCompletionRequest {
    fn messages(&self) -> Value {
        Value::from_serialize(&self.messages)
    }

    fn tools(&self) -> Option<Value> {
        if self.tools.is_none() {
            None
        } else {
            Some(Value::from_serialize(&self.tools))
        }
    }

    fn tool_choice(&self) -> Option<Value> {
        if self.tool_choice.is_none() {
            None
        } else {
            Some(Value::from_serialize(&self.tool_choice))
        }
    }

    fn should_add_generation_prompt(&self) -> bool {
        if let Some(last) = self.messages.last() {
            last.role == MessageRole::user
        } else {
            true
        }
    }
}

impl OAIChatLikeRequest for CompletionRequest {
    fn messages(&self) -> Value {
        let message = ChatCompletionMessage {
            role: MessageRole::user,
            content: Content::Text(self.prompt.clone()),
            name: None,
        };
        Value::from_serialize(vec![message])
    }

    fn should_add_generation_prompt(&self) -> bool {
        true
    }
}

impl OAIPromptFormatter for HfTokenizerConfigJsonFormatter {
    fn supports_add_generation_prompt(&self) -> bool {
        self.supports_add_generation_prompt
    }

    fn render(&self, req: &dyn OAIChatLikeRequest) -> Result<String> {
        let mixins = Value::from_dyn_object(self.mixins.clone());

        let tools = req.tools();
        let has_tools = tools.is_some();
        let add_generation_prompt = req.should_add_generation_prompt();

        tracing::trace!(
            "Rendering prompt with tools: {:?}, add_generation_prompt: {}",
            has_tools,
            add_generation_prompt
        );

        let ctx = context! {
            messages => req.messages(),
            tools => tools,
            bos_token => self.config.bos_tok(),
            eos_token => self.config.eos_tok(),
            unk_token => self.config.unk_tok(),
            add_generation_prompt => add_generation_prompt,
            ..mixins
        };

        let ctx = context! { ..ctx, ..context! {

        }};

        let tmpl = if has_tools {
            self.env.get_template("tool_use")?
        } else {
            self.env.get_template("default")?
        };

        Ok(tmpl.render(&ctx)?)
    }
}
