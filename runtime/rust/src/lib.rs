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

//! Triton

#![allow(dead_code)]
#![allow(unused_imports)]

use std::sync::{Arc, Mutex};

pub use anyhow::{anyhow as error, Context as ErrorContext, Error, Ok as OK, Result};

use async_once_cell::OnceCell;
use tracing as log;

mod config;
pub use config::RuntimeConfig;

pub mod component;
pub mod discovery;
pub mod engine;
pub mod pipeline;
pub mod protocols;
pub mod runtime;
pub mod service;
pub mod transports;
pub mod worker;

pub mod distributed;

pub use tokio_util::sync::CancellationToken;
pub use worker::Worker;

/// Types of Tokio runtimes that can be used to construct a Triton [Runtime].
#[derive(Clone)]
enum RuntimeType {
    Shared(Arc<tokio::runtime::Runtime>),
    External(tokio::runtime::Handle),
}

/// Local [Runtime] which provides access to shared resources local to the physical node/machine.
#[derive(Debug, Clone)]
pub struct Runtime {
    id: Arc<String>,
    primary: RuntimeType,
    secondary: Arc<tokio::runtime::Runtime>,
    cancellation_token: CancellationToken,
}

/// Distributed [Runtime] which provides access to shared resources across the cluster, this includes
/// communication protocols and transports.
#[derive(Clone)]
pub struct DistributedRuntime {
    // local runtime
    runtime: Runtime,

    // we might consider a unifed transport manager here
    etcd_client: transports::etcd::Client,
    nats_client: transports::nats::Client,
    tcp_server: Arc<OnceCell<Arc<transports::tcp::server::TcpStreamServer>>>,

    // local registery for components
    // the registry allows us to use share runtime resources across instances of the same component object.
    // take fo example two instances of a client to the same remote component. The registry allows us to use
    // a single endpoint watcher for both clients, this keeps the number background tasking watching specific
    // paths in etcd to a minimum.
    component_registry: component::Registry,
}
