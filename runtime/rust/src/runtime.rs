/*
 * Copyright 2024-2025 NVIDIA CORPORATION & AFFILIATES
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License. You may obtain a copy of
 * the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

//! The [Runtime] module is the interface for [crate::component::Component][crate::component::Component]
//! to access shared resources. These include thread pool, memory allocators and other shared resources.
//!
//! The [Runtime] holds the primary [CancellationToken] which can be used to terminate all attached
//! [crate::component::Component][crate::component::Component].
//!
//! We expect in the future to offer topologically aware thread and memory resources, but for now the
//! set of resources is limited to the thread pool and cancellation token.
//!
//! Notes: We will need to do an evaluation on what is fully public, what is pub(crate) and what is
//! private; however, for now we are exposing most objects as fully public while the API is maturing.

use super::{error, log, Result, Runtime, RuntimeType};
use crate::config::{self, RuntimeConfig};

use futures::Future;
use once_cell::sync::OnceCell;
use std::sync::{Arc, Mutex};
use tokio::{signal, task::JoinHandle};

pub use tokio_util::sync::CancellationToken;

impl Runtime {
    fn new(runtime: RuntimeType) -> Result<Runtime> {
        // worker id
        let id = Arc::new(uuid::Uuid::new_v4().to_string());

        // create a cancellation token
        let cancellation_token = CancellationToken::new();

        // secondary runtime for background ectd/nats tasks
        let secondary = RuntimeConfig::single_threaded().create_runtime()?;

        Ok(Runtime {
            id,
            primary: runtime,
            secondary: Arc::new(secondary),
            cancellation_token,
        })
    }

    pub fn from_handle(handle: tokio::runtime::Handle) -> Result<Runtime> {
        let runtime = RuntimeType::External(handle);
        Runtime::new(runtime)
    }

    /// Create a [`Runtime`] instance from the settings
    /// See [`config::RuntimeConfig::from_settings`]
    pub fn from_settings() -> Result<Runtime> {
        let config = config::RuntimeConfig::from_settings()?;
        let owned = RuntimeType::Shared(Arc::new(config.create_runtime()?));
        Runtime::new(owned)
    }

    /// Create a [`Runtime`] with a single-threaded primary async tokio runtime
    pub fn single_threaded() -> Result<Runtime> {
        let config = config::RuntimeConfig::single_threaded();
        let owned = RuntimeType::Shared(Arc::new(config.create_runtime()?));
        Runtime::new(owned)
    }

    /// Returns the unique identifier for the [`Runtime`]
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Returns a [`tokio::runtime::Handle`] for the primary/application thread pool
    pub fn primary(&self) -> tokio::runtime::Handle {
        self.primary.handle()
    }

    /// Returns a [`tokio::runtime::Handle`] for the secondary/background thread pool
    pub fn secondary(&self) -> &Arc<tokio::runtime::Runtime> {
        &self.secondary
    }

    /// Access the primary [`CancellationToken`] for the [`Runtime`]
    pub fn primary_token(&self) -> CancellationToken {
        self.cancellation_token.clone()
    }

    /// Creates a child [`CancellationToken`] tied to the lifecycle of the [`Runtime`]'s root [`CancellationToken::child_token`] method.
    pub fn child_token(&self) -> CancellationToken {
        self.cancellation_token.child_token()
    }

    /// Shuts down the [`Runtime`] instance
    pub fn shutdown(&self) {
        self.cancellation_token.cancel();
    }
}

impl RuntimeType {
    /// Get [`tokio::runtime::Handle`] to runtime
    pub fn handle(&self) -> tokio::runtime::Handle {
        match self {
            RuntimeType::External(rt) => rt.clone(),
            RuntimeType::Shared(rt) => rt.handle().clone(),
        }
    }
}

impl std::fmt::Debug for RuntimeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RuntimeType::External(_) => write!(f, "RuntimeType::External"),
            RuntimeType::Shared(_) => write!(f, "RuntimeType::Shared"),
        }
    }
}
