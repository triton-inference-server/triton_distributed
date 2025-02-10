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

//! The [Component] module defines the top-level API for building distributed applications.
//!
//! A distributed application consists of a set of [Component] that can host one
//! or more [Endpoint]. Each [Endpoint] is a network-accessible service
//! that can be accessed by other [Component] in the distributed application.
//!
//! A [Component] is made discoverable by registering it with the distributed runtime under
//! a [`Namespace`].
//!
//! A [`Namespace`] is a logical grouping of [Component] that are grouped together.
//!
//! We might extend namespace to include grouping behavior, which would define groups of
//! components that are tightly coupled.
//!
//! A [Component] is the core building block of a distributed application. It is a logical
//! unit of work such as a `Preprocessor` or `SmartRouter` that has a well-defined role in the
//! distributed application.
//!
//! A [Component] can present to the distributed application one or more configuration files
//! which define how that component was constructed/configured and what capabilities it can
//! provide.
//!
//! Other [Component] can write to watching locations within a [Component] etcd
//! path. This allows the [Component] to take dynamic actions depending on the watch
//! triggers.
//!
//! TODO: Top-level Overview of Endpoints/Functions

use crate::discovery::Lease;

use super::{error, transports::nats::Slug, DistributedRuntime, Result};

use crate::pipeline::network::{ingress::push_endpoint::PushEndpoint, PushWorkHandler};
use async_nats::{
    rustls::quic,
    service::{Service, ServiceExt},
};
use derive_builder::Builder;
use derive_getters::Getters;
use educe::Educe;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc};
use validator::{Validate, ValidationError};

mod client;
mod endpoint;
mod registry;
mod service;

pub use client::Client;

#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum TransportType {
    NatsTcp(String),
}

#[derive(Clone)]
pub struct Registry {
    services: Arc<tokio::sync::Mutex<HashMap<String, Service>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentEndpointInfo {
    pub component: String,
    pub endpoint: String,
    pub namespace: String,
    pub lease_id: i64,
    pub transport: TransportType,
}

/// A [Component] a discoverable entity in the distributed runtime.
/// You can host [Endpoint] on a [Component] by first creating
/// a [Service] then adding one or more [Endpoint] to the [Service].
///
/// You can also issue a request to a [Component]'s [Endpoint] by creating a [Client].
#[derive(Educe, Builder, Clone, Validate)]
#[educe(Debug)]
#[builder(pattern = "owned")]
pub struct Component {
    #[builder(private)]
    #[educe(Debug(ignore))]
    drt: DistributedRuntime,

    /// Name of the component
    #[builder(setter(into))]
    #[validate(custom(function = "validate_allowed_chars"))]
    name: String,

    /// Namespace
    #[builder(setter(into))]
    #[validate(custom(function = "validate_allowed_chars"))]
    namespace: String,
}

impl Component {
    pub fn etcd_path(&self) -> String {
        format!("{}/components/{}", self.namespace, self.name)
    }

    fn slug(&self) -> Slug {
        Slug::from_string(self.etcd_path())
    }

    pub fn endpoint(&self, endpoint: impl Into<String>) -> Result<Endpoint> {
        let endpoint = Endpoint {
            component: self.clone(),
            name: endpoint.into(),
        };

        endpoint.validate()?;

        Ok(endpoint)
    }

    /// Get keys from etcd on the slug, splitting the endpoints and only returning the
    /// set of unique endpoints.
    pub async fn list_endpoints(&self) -> Vec<Endpoint> {
        unimplemented!("endpoints")
    }

    /// TODO
    ///
    /// This method will scrape the stats for all available services
    /// Returns a stream of `ServiceInfo` objects.
    /// This should be consumed by a `[tokio::time::timeout_at`] because each services
    /// will only respond once, but there is no way to know when all services have responded.
    pub async fn stats_stream(&self) -> Result<()> {
        unimplemented!("collect_stats")
    }

    pub fn service_builder(&self) -> service::ServiceConfigBuilder {
        service::ServiceConfigBuilder::from_component(self.clone())
    }
}

impl ComponentBuilder {
    pub fn from_runtime(drt: DistributedRuntime) -> Self {
        Self::default().drt(drt)
    }
}

#[derive(Debug, Clone, Validate)]
pub struct Endpoint {
    component: Component,

    /// Endpoint name
    #[validate(custom(function = "validate_allowed_chars"))]
    name: String,
}

impl Endpoint {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn etcd_path(&self) -> String {
        format!("{}/{}", self.component.etcd_path(), self.name)
    }

    pub fn etcd_path_with_id(&self, lease_id: i64) -> String {
        format!("{}:{:x}", self.etcd_path(), lease_id)
    }

    pub fn name_with_id(&self, lease_id: i64) -> String {
        format!("{}-{:x}", self.name, lease_id)
    }

    pub fn subject(&self, lease_id: i64) -> String {
        format!("{}.{}", self.component.slug(), self.name_with_id(lease_id))
    }

    pub async fn client<Req, Resp>(&self) -> Result<client::Client<Req, Resp>>
    where
        Req: Serialize + Send + Sync + 'static,
        Resp: for<'de> Deserialize<'de> + Send + Sync + 'static,
    {
        client::Client::new(self.clone()).await
    }

    pub fn endpoint_builder(&self) -> endpoint::EndpointConfigBuilder {
        endpoint::EndpointConfigBuilder::from_endpoint(self.clone())
    }
}

#[derive(Educe, Builder, Clone, Validate)]
#[educe(Debug)]
#[builder(pattern = "owned")]
pub struct Namespace {
    #[builder(private)]
    #[educe(Debug(ignore))]
    runtime: DistributedRuntime,

    #[validate(custom(function = "validate_allowed_chars"))]
    name: String,
}

impl Namespace {
    pub(crate) fn new(runtime: DistributedRuntime, name: String) -> Result<Self> {
        let namespace = NamespaceBuilder::default()
            .runtime(runtime)
            .name(name)
            .build()?;

        namespace.validate()?;

        Ok(namespace)
    }

    /// Create a [`Component`] in the namespace
    pub fn component(&self, name: impl Into<String>) -> Result<Component> {
        let component = ComponentBuilder::from_runtime(self.runtime.clone())
            .name(name)
            .namespace(self.name.clone())
            .build()?;

        component.validate()?;

        Ok(component)
    }
}

// Custom validator function
fn validate_allowed_chars(input: &str) -> Result<(), ValidationError> {
    // Define the allowed character set using a regex
    let regex = regex::Regex::new(r"^[a-z0-9_]+(?:-+[a-z0-9_]+)*$").unwrap();

    if regex.is_match(input) {
        Ok(())
    } else {
        Err(ValidationError::new("invalid_characters"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use validator::Validate;

    // Mock struct used for testing validation logic.
    #[derive(Validate)]
    struct InputData {
        #[validate(custom(function = "validate_allowed_chars"))]
        name: String,
    }

    #[test]
    fn test_valid_names() {
        // Valid strings
        let valid_inputs = vec![
            "abc",       // Lowercase letters
            "abc123",    // Letters and numbers
            "a-b-c",     // Letters with hyphens
            "a_b_c",     // Letters with underscores
            "a-b_c-123", // Mixed valid characters
            "a",         // Single character
            "a_b",       // Short valid pattern
            "123456",    // Only numbers
        ];

        for input in valid_inputs {
            let result = validate_allowed_chars(input);
            assert!(result.is_ok(), "Expected '{}' to be valid", input);
        }
    }

    #[test]
    fn test_invalid_names() {
        // Invalid strings
        let invalid_inputs = vec![
            "abc!",     // Invalid character `!`
            "abc@",     // Invalid character `@`
            "123$",     // Invalid character `$`
            "foo.bar",  // Invalid character `.`
            "foo/bar",  // Invalid character `/`
            "foo\\bar", // Invalid character `\`
            "abc#",     // Invalid character `#`
            "abc def",  // Spaces are not allowed
            "foo,",     // Invalid character `,`
            "",         // Empty string
        ];

        for input in invalid_inputs {
            let result = validate_allowed_chars(input);
            assert!(result.is_err(), "Expected '{}' to be invalid", input);
        }
    }

    #[test]
    fn test_struct_validation_valid() {
        // Struct with valid data
        let valid_data = InputData {
            name: "valid-name_123".to_string(),
        };
        assert!(valid_data.validate().is_ok());
    }

    #[test]
    fn test_struct_validation_invalid() {
        // Struct with invalid data
        let invalid_data = InputData {
            name: "invalid!name".to_string(),
        };
        let result = invalid_data.validate();
        assert!(result.is_err());

        if let Err(errors) = result {
            let error_map = errors.field_errors();
            assert!(error_map.contains_key("name"));
            let name_errors = &error_map["name"];
            assert_eq!(name_errors[0].code, "invalid_characters");
        }
    }

    #[test]
    fn test_edge_cases() {
        // Edge cases
        let edge_inputs = vec![
            ("-", false),  // Single hyphen
            ("_", true),   // Single underscore
            ("a-", false), // Letter with hyphen
            ("--", false), // Repeated hyphens
            ("-a", false), // Hyphen at the beginning
            ("a-", false), // Hyphen at the end
        ];

        for (input, expected_validity) in edge_inputs {
            let result = validate_allowed_chars(input);
            if expected_validity {
                assert!(result.is_ok(), "Expected '{}' to be valid", input);
            } else {
                assert!(result.is_err(), "Expected '{}' to be invalid", input);
            }
        }
    }
}
