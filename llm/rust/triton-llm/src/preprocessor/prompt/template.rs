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

use std::{collections::HashSet, sync::Arc};

use anyhow::{Ok, Result};
use minijinja::Environment;

use crate::model_card::model::{ModelDeploymentCard, PromptContextMixin, PromptFormatterArtifact};

mod context;
mod formatters;
mod oai;
mod tokcfg;

use super::{OAIChatLikeRequest, OAIPromptFormatter, PromptFormatter};
use tokcfg::{raise_exception, tojson, ChatTemplate as HfTokenizerConfig};

impl PromptFormatter {
    pub async fn from_mdc(mdc: ModelDeploymentCard) -> Result<PromptFormatter> {
        match mdc
            .prompt_formatter
            .ok_or(anyhow::anyhow!("MDC does not contain a prompt formatter"))?
        {
            PromptFormatterArtifact::HfTokenizerConfigJson(file) => {
                let content = std::fs::read_to_string(file)?;
                let config: HfTokenizerConfig = serde_json::from_str(&content)?;
                let formatter = HfTokenizerConfigJsonFormatter::new(
                    config,
                    mdc.prompt_context
                        .map_or(ContextMixins::default(), |x| ContextMixins::new(&x)),
                )?;
                Ok(Self::OAI(Arc::new(formatter)))
            }
        }
    }
}

/// Chat Template Jinja Renderer
///
/// This object holds a Jinja environment with at least one registered template.
///
/// If [`ChatTemplateValue`] is a string, it is registered as the `default` template.
///
/// If [`ChatTemplateValue`] is a map, it is expected to contain at least one of the following keys:
/// - `tool_use`: The template to use when a tool is used.
/// - `default`: The template to use when no tool is used.
///
/// If the map contains both keys, the `tool_use` template is registered as the `tool_use` template
/// and the `default` template is registered as the `default` template.
///
struct JinjaEnvironment {
    env: Environment<'static>,
}

#[derive(Debug)]
struct HfTokenizerConfigJsonFormatter {
    env: Environment<'static>,
    config: HfTokenizerConfig,
    mixins: Arc<ContextMixins>,
    supports_add_generation_prompt: bool,
}

// /// OpenAI Standard Prompt Formatter
// pub trait StandardPromptFormatter {
//     fn render(&self, context: &impl StandardPromptContext) -> Result<String>;
// }

// pub trait StandardPromptContext {
//     fn messages(&self) -> Value;
//     fn tools(&self) -> Option<Value>;
// }

#[derive(Debug, Clone, Default)]
pub struct ContextMixins {
    context_mixins: HashSet<PromptContextMixin>,
}
