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

pub mod annotated;

pub type LeaseId = i64;

#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct Component {
    pub name: String,
    pub namespace: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct Endpoint {
    /// Name of the endpoint.
    pub name: String,

    /// Component of the endpoint.
    pub component: Component,

    /// Namespace of the component.
    pub namespace: String,

    /// Optional lease id for the endpoint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lease: Option<LeaseId>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum RouterType {
    PushRoundRobin,
    PushRandom,
}

impl Default for RouterType {
    fn default() -> Self {
        Self::PushRandom
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct ModelMetaData {
    pub name: String,
    pub component: Component,
    pub router_type: RouterType,
}
