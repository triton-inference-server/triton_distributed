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
use std::convert::TryFrom;
use std::str::FromStr;

use crate::pipeline::PipelineError;

pub mod annotated;

pub type LeaseId = i64;

#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct Component {
    pub name: String,
    pub namespace: String,
}

/// Represents an endpoint with a namespace, component, and name.
///
/// An `Endpoint` is defined by a three-part string separated by `/`:
/// - **namespace**
/// - **component**
/// - **name**
///
/// Example format: `"namespace/component/endpoint"`
#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct Endpoint {
    /// Name of the endpoint.
    pub name: String,

    /// Component of the endpoint.
    pub component: String,

    /// Namespace of the component.
    pub namespace: String,
}

impl TryFrom<&str> for Endpoint {
    type Error = PipelineError;

    /// Attempts to create an `Endpoint` from a string.
    ///
    /// # Arguments
    /// - `path`: A string in the format `"namespace/component/endpoint"`.
    ///
    /// # Errors
    /// Returns a `PipelineError::InvalidFormat` if the input string does not
    /// have exactly three parts separated by `/`.
    ///
    /// # Examples
    /// ```ignore
    /// use std::convert::TryFrom;
    /// use triton_distributed::protocols::Endpoint;
    ///
    /// let endpoint = Endpoint::try_from("namespace/component/endpoint").unwrap();
    /// assert_eq!(endpoint.namespace, "namespace");
    /// assert_eq!(endpoint.component, "component");
    /// assert_eq!(endpoint.name, "endpoint");
    /// ```
    fn try_from(path: &str) -> Result<Self, Self::Error> {
        let elements: Vec<&str> = path.split('/').collect();
        if elements.len() != 3 {
            return Err(PipelineError::InvalidEndpointFormat);
        }

        Ok(Endpoint {
            namespace: elements[0].to_string(),
            component: elements[1].to_string(),
            name: elements[2].to_string(),
        })
    }
}

impl FromStr for Endpoint {
    type Err = PipelineError;

    /// Parses an `Endpoint` from a string using the standard Rust `.parse::<T>()` pattern.
    ///
    /// This is implemented in terms of [`TryFrom<&str>`].
    ///
    /// # Errors
    /// Returns an `PipelineError::InvalidFormat` if the input does not match `"namespace/component/endpoint"`.
    ///
    /// # Examples
    /// ```ignore
    /// use std::str::FromStr;
    /// use triton_distributed::protocols::Endpoint;
    ///
    /// let endpoint: Endpoint = "namespace/component/endpoint".parse().unwrap();
    /// assert_eq!(endpoint.namespace, "namespace");
    /// assert_eq!(endpoint.component, "component");
    /// assert_eq!(endpoint.name, "endpoint");
    /// ```
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Endpoint::try_from(s)
    }
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
    use std::convert::TryFrom;
    use std::str::FromStr;

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

    #[test]
    fn test_valid_endpoint_try_from() {
        let input = "namespace1/component1/endpoint1";
        let endpoint = Endpoint::try_from(input).expect("Valid endpoint should parse successfully");

        assert_eq!(endpoint.namespace, "namespace1");
        assert_eq!(endpoint.component, "component1");
        assert_eq!(endpoint.name, "endpoint1");
    }

    #[test]
    fn test_valid_endpoint_from_str() {
        let input = "namespace2/component2/endpoint2";
        let endpoint = Endpoint::from_str(input).expect("Valid endpoint should parse successfully");

        assert_eq!(endpoint.namespace, "namespace2");
        assert_eq!(endpoint.component, "component2");
        assert_eq!(endpoint.name, "endpoint2");
    }

    #[test]
    fn test_valid_endpoint_parse() {
        let input = "namespace3/component3/endpoint3";
        let endpoint: Endpoint = input
            .parse()
            .expect("Valid endpoint should parse successfully");

        assert_eq!(endpoint.namespace, "namespace3");
        assert_eq!(endpoint.component, "component3");
        assert_eq!(endpoint.name, "endpoint3");
    }

    #[test]
    fn test_invalid_endpoint_try_from() {
        let input = "invalid_endpoint_format";

        let result = Endpoint::try_from(input);
        assert!(result.is_err(), "Parsing should fail for an invalid format");
        assert_eq!(
            result.unwrap_err().to_string(),
            "An endpoint URL must have the format: namespace/component/endpoint"
        );
    }

    #[test]
    fn test_invalid_endpoint_from_str() {
        let input = "onlyhas/two";

        let result = Endpoint::from_str(input);
        assert!(result.is_err(), "Parsing should fail for an invalid format");
        assert_eq!(
            result.unwrap_err().to_string(),
            "An endpoint URL must have the format: namespace/component/endpoint"
        );
    }

    #[test]
    fn test_invalid_endpoint_parse() {
        let input = "too/many/segments/in/url";

        let result: Result<Endpoint, _> = input.parse();
        assert!(result.is_err(), "Parsing should fail for an invalid format");
        assert_eq!(
            result.unwrap_err().to_string(),
            "An endpoint URL must have the format: namespace/component/endpoint"
        );
    }

    #[test]
    fn test_empty_endpoint_string() {
        let input = "";

        let result = Endpoint::try_from(input);
        assert!(result.is_err(), "Parsing should fail for an empty string");
        assert_eq!(
            result.unwrap_err().to_string(),
            "An endpoint URL must have the format: namespace/component/endpoint"
        );
    }

    #[test]
    fn test_whitespace_endpoint_string() {
        let input = "   ";

        let result = Endpoint::try_from(input);
        assert!(
            result.is_err(),
            "Parsing should fail for a whitespace string"
        );
        assert_eq!(
            result.unwrap_err().to_string(),
            "An endpoint URL must have the format: namespace/component/endpoint"
        );
    }

    #[test]
    fn test_leading_trailing_slashes() {
        let input = "/namespace/component/endpoint/";

        let result = Endpoint::try_from(input);
        assert!(
            result.is_err(),
            "Parsing should fail for leading/trailing slashes"
        );
    }
}
