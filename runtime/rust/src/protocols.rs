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
    pub component: String,

    /// Namespace of the component.
    pub namespace: String,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_component_creation() {
        let component = Component {
            name: "test_name".to_string(),
            namespace: "test_namespace".to_string(),
        };

        assert_eq!(component.name, "test_name");
        assert_eq!(component.namespace, "test_namespace");
    }

    #[test]
    fn test_endpoint_creation() {
        let endpoint = Endpoint {
            name: "test_endpoint".to_string(),
            component: "test_component".to_string(),
            namespace: "test_namespace".to_string(),
        };

        assert_eq!(endpoint.name, "test_endpoint");
        assert_eq!(endpoint.component, "test_component");
        assert_eq!(endpoint.namespace, "test_namespace");
    }

    #[test]
    fn test_router_type_default() {
        let default_router = RouterType::default();
        assert_eq!(default_router, RouterType::PushRandom);
    }

    #[test]
    fn test_router_type_serialization() {
        let router_round_robin = RouterType::PushRoundRobin;
        let router_random = RouterType::PushRandom;

        let serialized_round_robin = serde_json::to_string(&router_round_robin).unwrap();
        let serialized_random = serde_json::to_string(&router_random).unwrap();

        assert_eq!(serialized_round_robin, "\"push_round_robin\"");
        assert_eq!(serialized_random, "\"push_random\"");
    }

    #[test]
    fn test_router_type_deserialization() {
        let round_robin: RouterType = serde_json::from_str("\"push_round_robin\"").unwrap();
        let random: RouterType = serde_json::from_str("\"push_random\"").unwrap();

        assert_eq!(round_robin, RouterType::PushRoundRobin);
        assert_eq!(random, RouterType::PushRandom);
    }

    #[test]
    fn test_model_metadata_creation() {
        let component = Component {
            name: "test_component".to_string(),
            namespace: "test_namespace".to_string(),
        };

        let metadata = ModelMetaData {
            name: "test_model".to_string(),
            component,
            router_type: RouterType::PushRoundRobin,
        };

        assert_eq!(metadata.name, "test_model");
        assert_eq!(metadata.component.name, "test_component");
        assert_eq!(metadata.component.namespace, "test_namespace");
        assert_eq!(metadata.router_type, RouterType::PushRoundRobin);
    }
}
