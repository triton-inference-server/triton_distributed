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

mod request;
mod response;

pub use request::*;
pub use response::*;
use serde_json::Value;
use std::collections::HashMap;
use uuid::Uuid;

/// Matches and processes tool calling patterns in LLM responses
///
/// Supports multiple formats for tool calls:
/// - Single/multiple function calls with parameters/arguments
/// - Auto or user selected tool usage
pub struct ToolCallingMatcher {
    tool_choice: ToolChoice,
}

// Same as CalledFunction with named parameters
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CalledFunctionParameters {
    pub name: String,
    pub parameters: HashMap<String, Value>,
}

// Same as CalledFunction with named parameters
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CalledFunctionArguments {
    pub name: String,
    pub arguments: HashMap<String, Value>,
}

impl ToolCallingMatcher {
    pub fn new(tool_choice: ToolChoice) -> anyhow::Result<Self> {
        Ok(Self { tool_choice })
    }

    pub fn get_call(&self, message: &str) -> anyhow::Result<Vec<ToolCallResponse>> {
        if matches!(self.tool_choice, ToolChoice::None) {
            return Ok(Vec::new());
        }

        if let Ok(deser) = serde_json::from_str::<CalledFunctionParameters>(message) {
            let id = format!("call-{}", Uuid::new_v4());
            Ok(vec![ToolCallResponse {
                id,
                tp: ToolCallType::Function,
                function: CalledFunction {
                    name: deser.name,
                    arguments: serde_json::to_string(&deser.parameters)?,
                },
            }])
        } else if let Ok(deser) = serde_json::from_str::<Vec<CalledFunctionParameters>>(message) {
            Ok(deser
                .into_iter()
                .map(|deser| {
                    let id = format!("call-{}", Uuid::new_v4());
                    Ok(ToolCallResponse {
                        id,
                        tp: ToolCallType::Function,
                        function: CalledFunction {
                            name: deser.name,
                            arguments: serde_json::to_string(&deser.parameters)?,
                        },
                    })
                })
                .collect::<anyhow::Result<Vec<_>>>()?)
        } else if let Ok(deser) = serde_json::from_str::<CalledFunctionArguments>(message) {
            let id = format!("call-{}", Uuid::new_v4());
            Ok(vec![ToolCallResponse {
                id,
                tp: ToolCallType::Function,
                function: CalledFunction {
                    name: deser.name,
                    arguments: serde_json::to_string(&deser.arguments)?,
                },
            }])
        } else if let Ok(deser) = serde_json::from_str::<Vec<CalledFunctionArguments>>(message) {
            Ok(deser
                .into_iter()
                .map(|deser| {
                    let id = format!("call-{}", Uuid::new_v4());
                    Ok(ToolCallResponse {
                        id,
                        tp: ToolCallType::Function,
                        function: CalledFunction {
                            name: deser.name,
                            arguments: serde_json::to_string(&deser.arguments)?,
                        },
                    })
                })
                .collect::<anyhow::Result<Vec<_>>>()?)
        } else {
            if matches!(self.tool_choice, ToolChoice::Tool(_)) {
                anyhow::bail!("Tool choice was required but no tools were called.")
            }
            Ok(Vec::new())
        }
    }
}
