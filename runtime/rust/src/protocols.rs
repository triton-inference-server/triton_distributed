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

use crate::component::validate_allowed_chars;
use crate::{error, raise, DistributedRuntime, Error, Result};
use derive_getters::{Dissolve, Getters};
use serde::{Deserialize, Serialize};
use validator::ValidationError;

pub mod annotated;

pub type LeaseId = i64;

#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct Component {
    namespace: String,
    name: String,
}
#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq, Getters, Dissolve)]
pub struct Endpoint {
    /// Namespace of the component.
    namespace: String,

    /// Component of the endpoint.
    component: String,

    /// Name of the endpoint.
    name: String,
}

impl std::fmt::Display for Endpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.namespace, self.component, self.name)
    }
}

/// An [EndpointAddress] represents a fully qualified endpoint address in the format:
/// `namespace.component.endpoint`
/// Each segment must contain only lowercase letters, numbers, hyphens, and underscores.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EndpointAddress(String);

impl EndpointAddress {
    /// Creates a new EndpointAddress from namespace, component, and endpoint names
    pub fn new(namespace: &str, component: &str, endpoint: &str) -> Result<Self> {
        // Validate each segment
        validate_allowed_chars(namespace)
            .map_err(|_| error!("invalid namespace: {}", namespace))?;
        validate_allowed_chars(component)
            .map_err(|_| error!("invalid component: {}", component))?;
        validate_allowed_chars(endpoint).map_err(|_| error!("invalid endpoint: {}", endpoint))?;

        Ok(Self(format!("{}.{}.{}", namespace, component, endpoint)))
    }

    /// Returns the full address as a string
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::str::FromStr for EndpointAddress {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parts: Vec<&str> = s.split('.').collect();
        if parts.len() != 3 {
            raise!("invalid endpoint address format: {}", s);
        }

        Self::new(parts[0], parts[1], parts[2])
    }
}

impl std::fmt::Display for EndpointAddress {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl AsRef<str> for EndpointAddress {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl From<EndpointAddress> for Endpoint {
    fn from(addr: EndpointAddress) -> Self {
        let parts: Vec<&str> = addr.0.split('.').collect();
        Endpoint {
            namespace: parts[0].to_string(),
            component: parts[1].to_string(),
            name: parts[2].to_string(),
        }
    }
}

impl From<Endpoint> for EndpointAddress {
    fn from(endpoint: Endpoint) -> Self {
        // We can unwrap here because Endpoint should already have valid segments
        Self::new(&endpoint.namespace, &endpoint.component, &endpoint.name)
            .expect("Endpoint contained invalid segments")
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
    fn test_valid_endpoint_address() {
        let valid_cases = vec![
            ("ns1", "comp1", "ep1"),
            ("my-ns", "my-comp", "my-ep"),
            ("ns_1", "comp_1", "ep_1"),
            ("ns-1", "comp-1", "ep-1"),
        ];

        for (ns, comp, ep) in valid_cases {
            let addr = EndpointAddress::new(ns, comp, ep);
            assert!(addr.is_ok());
            let addr = addr.unwrap();
            assert_eq!(addr.to_string(), format!("{}.{}.{}", ns, comp, ep));
        }
    }

    #[test]
    fn test_invalid_endpoint_address() {
        let invalid_cases = vec![
            ("NS1", "comp1", "ep1"),    // uppercase not allowed
            ("my ns", "comp1", "ep1"),  // spaces not allowed
            ("my.ns", "comp1", "ep1"),  // dots not allowed in segments
            ("my@ns", "comp1", "ep1"),  // special chars not allowed
            ("my-ns", "_comp1", "ep1"), // leading underscore not allowed
            ("", "comp1", "ep1"),       // empty not allowed
        ];

        for (ns, comp, ep) in invalid_cases {
            let addr = EndpointAddress::new(ns, comp, ep);
            assert!(
                addr.is_err(),
                "Expected error for invalid address: {}.{}.{}",
                ns,
                comp,
                ep
            );
        }
    }

    #[test]
    fn test_parse_endpoint_address() {
        let valid = "ns1.comp1.ep1";
        let addr = valid.parse::<EndpointAddress>();
        assert!(addr.is_ok());
        assert_eq!(addr.unwrap().to_string(), valid);

        let invalid = "ns1.comp1"; // missing segment
        let addr = invalid.parse::<EndpointAddress>();
        assert!(addr.is_err());
    }

    #[test]
    fn test_endpoint_conversions() {
        let addr = EndpointAddress::new("ns1", "comp1", "ep1").unwrap();

        // Convert to Endpoint
        let endpoint: Endpoint = addr.clone().into();
        assert_eq!(endpoint.namespace, "ns1");
        assert_eq!(endpoint.component, "comp1");
        assert_eq!(endpoint.name, "ep1");

        // Convert back to EndpointAddress
        let converted_addr: EndpointAddress = endpoint.into();
        assert_eq!(converted_addr, addr);
    }
}
