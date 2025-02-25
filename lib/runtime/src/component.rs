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
//! [Components][Component] are addressed via a hierarchy of:
//! - [Namespace]
//! - [Component]
//! - [Function]
//!
//! A [Component] can host one or more [Functions][Function].
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
//!
//!
//! [Components'][Component] can be served/hosted by creating a [Service] and adding one or more
//! [Endpoints][Endpoint] to the [Service].
//!
//! A [Component's][Component] [Function] can be called by any member of the distributed runtime
//! by creating a [Client] bound to the [Component's][Component] [Endpoint][Endpoint].
//!
//! A distributed application consists of a set of [Component] that can host one
//! or more [Endpoint]. Each [Endpoint] is a network-accessible service
//! that can be accessed by other [Component] in the distributed application.
//! TODO: Top-level Overview of Endpoints/Functions

use crate::{discovery::Lease, service::ServiceSet};

use super::{
    error, traits::*, transports::nats::Slug, utils::Duration, DistributedRuntime, Result, Runtime,
};

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
mod function;
mod namespace;
mod registry;
mod service;

pub use client::Client;
pub use function::Function;
pub use namespace::Namespace;

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
    pub function: String,
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

    // todo - restrict the namespace to a-z0-9-_A-Z
    /// Name of the component
    #[builder(setter(into))]
    name: String,

    // todo - restrict the namespace to a-z0-9-_A-Z
    /// Namespace
    #[builder(setter(into))]
    #[validate(custom(function = "validate_allowed_chars"))]
    namespace: String,
}

impl DistributedRuntimeProvider for Component {
    fn drt(&self) -> &DistributedRuntime {
        &self.drt
    }
}

impl RuntimeProvider for Component {
    fn rt(&self) -> &Runtime {
        self.drt.rt()
    }
}

impl Component {
    pub fn etcd_path(&self) -> String {
        format!("{}/components/{}", self.namespace, self.name)
    }

    pub fn service_name(&self) -> String {
        Slug::from_string(format!("{}|{}", self.namespace, self.name)).to_string()
    }

    // todo - move to EventPlane
    pub fn event_subject(&self, name: impl AsRef<str>) -> String {
        format!("{}.events.{}", self.service_name(), name.as_ref())
    }

    pub fn path(&self) -> String {
        format!("{}/{}", self.namespace, self.name)
    }

    pub fn drt(&self) -> &DistributedRuntime {
        &self.drt
    }

    pub fn function(&self, name: impl Into<String>) -> Result<Function> {
        Function::new(self.clone(), name.into())
    }

    // /// Get keys from etcd on the slug, splitting the endpoints and only returning the
    // /// set of unique endpoints.
    // pub async fn list_function(&self) -> Vec<Function> {
    //     unimplemented!("endpoints")
    // }

    pub async fn scrape_stats(&self, duration: Duration) -> Result<ServiceSet> {
        let service_name = self.service_name();
        let service_client = self.drt().service_client();
        service_client
            .collect_services(&service_name, duration)
            .await
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

// Custom validator function
fn validate_allowed_chars(input: &str) -> Result<(), ValidationError> {
    // Define the allowed character set using a regex
    let regex = regex::Regex::new(r"^[a-z][a-z0-9-_]*[a-z0-9]$").unwrap();

    if regex.is_match(input) {
        Ok(())
    } else {
        Err(ValidationError::new("invalid_characters"))
    }
}

// TODO - enable restrictions to the character sets allowed for namespaces,
// components, and endpoints.
//
// Put Validate traits on the struct and use the `validate_allowed_chars` method
// to validate the fields.

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use validator::Validate;

//     #[test]
//     fn test_valid_names() {
//         // Valid strings
//         let valid_inputs = vec![
//             "abc",        // Lowercase letters
//             "abc123",     // Letters and numbers
//             "a-b-c",      // Letters with hyphens
//             "a_b_c",      // Letters with underscores
//             "a-b_c-123",  // Mixed valid characters
//             "a",          // Single character
//             "a_b",        // Short valid pattern
//             "123456",     // Only numbers
//             "a---b_c123", // Repeated hyphens/underscores
//         ];

//         for input in valid_inputs {
//             let result = validate_allowed_chars(input);
//             assert!(result.is_ok(), "Expected '{}' to be valid", input);
//         }
//     }

//     #[test]
//     fn test_invalid_names() {
//         // Invalid strings
//         let invalid_inputs = vec![
//             "abc!",     // Invalid character `!`
//             "abc@",     // Invalid character `@`
//             "123$",     // Invalid character `$`
//             "foo.bar",  // Invalid character `.`
//             "foo/bar",  // Invalid character `/`
//             "foo\\bar", // Invalid character `\`
//             "abc#",     // Invalid character `#`
//             "abc def",  // Spaces are not allowed
//             "foo,",     // Invalid character `,`
//             "",         // Empty string
//         ];

//         for input in invalid_inputs {
//             let result = validate_allowed_chars(input);
//             assert!(result.is_err(), "Expected '{}' to be invalid", input);
//         }
//     }

//     // #[test]
//     // fn test_struct_validation_valid() {
//     //     // Struct with valid data
//     //     let valid_data = InputData {
//     //         name: "valid-name_123".to_string(),
//     //     };
//     //     assert!(valid_data.validate().is_ok());
//     // }

//     // #[test]
//     // fn test_struct_validation_invalid() {
//     //     // Struct with invalid data
//     //     let invalid_data = InputData {
//     //         name: "invalid!name".to_string(),
//     //     };
//     //     let result = invalid_data.validate();
//     //     assert!(result.is_err());

//     //     if let Err(errors) = result {
//     //         let error_map = errors.field_errors();
//     //         assert!(error_map.contains_key("name"));
//     //         let name_errors = &error_map["name"];
//     //         assert_eq!(name_errors[0].code, "invalid_characters");
//     //     }
//     // }

//     #[test]
//     fn test_edge_cases() {
//         // Edge cases
//         let edge_inputs = vec![
//             ("-", true),   // Single hyphen
//             ("_", true),   // Single underscore
//             ("a-", true),  // Letter with hyphen
//             ("-", false),  // Repeated hyphens
//             ("-a", false), // Hyphen at the beginning
//             ("a-", false), // Hyphen at the end
//         ];

//         for (input, expected_validity) in edge_inputs {
//             let result = validate_allowed_chars(input);
//             if expected_validity {
//                 assert!(result.is_ok(), "Expected '{}' to be valid", input);
//             } else {
//                 assert!(result.is_err(), "Expected '{}' to be invalid", input);
//             }
//         }
//     }
// }
