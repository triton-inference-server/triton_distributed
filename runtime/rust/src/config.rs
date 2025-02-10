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

use super::Result;
use derive_builder::Builder;
use figment::{
    providers::{Env, Format, Serialized, Toml},
    Figment,
};
use serde::{Deserialize, Serialize};
use validator::Validate;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerConfig {
    /// Grace shutdown period for http-service.
    pub graceful_shutdown_timeout: u64,
}

impl WorkerConfig {
    pub fn from_settings() -> Self {
        // Instantiates and reads server configurations from appropriate sources.
        // All calls should be global and thread safe.
        Figment::new()
            .merge(Serialized::defaults(Self::default()))
            .merge(Env::prefixed("TRITON_WORKER_"))
            .extract()
            .unwrap()
    }
}

impl Default for WorkerConfig {
    fn default() -> Self {
        WorkerConfig {
            graceful_shutdown_timeout: if cfg!(debug_assertions) {
                1 // Debug build: 1 second
            } else {
                30 // Release build: 30 seconds
            },
        }
    }
}

/// Runtime configuration
/// Defines the configuration for Tokio runtimes
#[derive(Serialize, Deserialize, Validate, Debug, Builder, Clone)]
#[builder(build_fn(private, name = "build_internal"), derive(Debug, Serialize))]
pub struct RuntimeConfig {
    /// Maximum number of async worker threads
    /// If set to 1, the runtime will run in single-threaded mode
    #[validate(range(min = 1))]
    #[builder(default = "16")]
    #[builder_field_attr(serde(skip_serializing_if = "Option::is_none"))]
    pub max_worker_threads: usize,

    /// Maximum number of blocking threads
    /// Blocking threads are used for blocking operations, this value must be greater than 0.
    #[validate(range(min = 1))]
    #[builder(default = "512")]
    #[builder_field_attr(serde(skip_serializing_if = "Option::is_none"))]
    pub max_blocking_threads: usize,
}

impl RuntimeConfig {
    pub fn builder() -> RuntimeConfigBuilder {
        RuntimeConfigBuilder::default()
    }

    pub(crate) fn figment() -> Figment {
        Figment::new()
            .merge(Serialized::defaults(RuntimeConfig::default()))
            .merge(Toml::file("/opt/triton/defaults/runtime.toml"))
            .merge(Toml::file("/opt/triton/etc/runtime.toml"))
            .merge(Env::prefixed("TRITON_RUNTIME_"))
    }

    /// Load the runtime configuration from the environment and configuration files
    /// Configuration is priorities in the following order, where the last has the lowest priority:
    /// 1. Environment variables (top priority)
    /// 2. /opt/triton/etc/runtime.toml
    /// 3. /opt/triton/defaults/runtime.toml (lowest priority)
    ///
    /// Environment variables are prefixed with `TRITON_RUNTIME_`
    pub fn from_settings() -> Result<RuntimeConfig> {
        let config: RuntimeConfig = Self::figment().extract()?;
        config.validate()?;
        Ok(config)
    }

    pub fn single_threaded() -> Self {
        RuntimeConfig {
            max_worker_threads: 1,
            max_blocking_threads: 1,
        }
    }

    /// Create a new default runtime configuration
    pub(crate) fn create_runtime(&self) -> Result<tokio::runtime::Runtime> {
        Ok(tokio::runtime::Builder::new_multi_thread()
            .worker_threads(self.max_worker_threads)
            .max_blocking_threads(self.max_blocking_threads)
            .enable_all()
            .build()?)
    }
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            max_worker_threads: 16,
            max_blocking_threads: 16,
        }
    }
}

impl RuntimeConfigBuilder {
    /// Build and validate the runtime configuration
    pub fn build(&self) -> Result<RuntimeConfig> {
        let config = self.build_internal()?;
        config.validate()?;
        Ok(config)
    }
}
