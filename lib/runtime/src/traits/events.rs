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

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

use crate::Result;

// #[async_trait]
// pub trait Publisher: Debug + Clone + Send + Sync {
//     async fn publish(&self, event: &(impl Serialize + Send + Sync)) -> Result<()>;
// }

/// A [EventPlane] is a component that can publish and/or subscribe to events.
///
/// Each implementation of [EventPlane] will define the root subject.
#[async_trait]
pub trait EventPublisher {
    /// The base subject used for this implementation of the [EventPlane].
    fn subject(&self) -> String;

    /// Publish a single event to the event plane. The `event_name` will be `.` concatenated with the
    /// base subject provided by the implementation.
    async fn publish(
        &self,
        event_name: impl AsRef<str> + Send + Sync,
        event: &(impl Serialize + Send + Sync),
    ) -> Result<()>;

    /// Publish a single event as bytes to the event plane. The `event_name` will be `.` concatenated with the
    /// base subject provided by the implementation.
    async fn publish_bytes(
        &self,
        event_name: impl AsRef<str> + Send + Sync,
        bytes: Vec<u8>,
    ) -> Result<()>;

    // /// Create a new publisher for the given event name. The `event_name` will be `.` concatenated with the
    // /// base subject provided by the implementation.
    // fn publisher(&self, event_name: impl AsRef<str>) -> impl Publisher;

    // /// Create a new publisher for the given event name. The `event_name` will be `.` concatenated with the
    // fn publisher(&self, event_name: impl AsRef<str>) -> Result<Publisher>;
    // fn publisher_bytes(&self, event_name: impl AsRef<str>) -> &PublisherBytes;
}
