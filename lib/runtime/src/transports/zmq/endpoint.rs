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

//! ZMQ Endpoint
//!
//! A ZMQ endpoint is a [Dealer] socket that is used to kick off an AsyncEngine / Pipeline
//! with request, then propagate the responses.
//!

use std::{future::Future, sync::Arc, vec::IntoIter};

use async_zmq::{zmq, Dealer};
use tokio::{sync::mpsc, task};
use tokio_util::sync::CancellationToken;

use derive_builder::Builder;

use crate::{
    component::Endpoint,
    engine::AsyncEngine,
    pipeline::{ManyOut, SingleIn},
    raise, Error, Result,
};

pub enum ZmqEndpointOperation {}

#[derive(Builder)]
pub struct ZmqEndpointConfig {
    endpoint: Endpoint,

    // if some, executes the engine
    // if none, reroutes requests to `ipc-` and proxies back the responses to the stream.
    ingress: Option<Arc<dyn AsyncEngine<SingleIn<Vec<u8>>, ManyOut<Vec<u8>>, Error>>>,
}

pub struct ZmqEndpoint {
    ops_tx: mpsc::Sender<ZmqEndpointOperation>,
}

impl ZmqEndpoint {}

pub struct TaskExecutionHandle {
    cancel_token: CancellationToken,
    handle: task::JoinHandle<Result<()>>,
}

impl TaskExecutionHandle {
    pub fn new<State, F, Fut>(
        state: State,
        future: F,
        cancel_token: CancellationToken,
        description: &str,
    ) -> Result<Self>
    where
        State: Send + 'static,
        F: FnOnce(State, CancellationToken) -> Fut + Send + 'static,
        Fut: Future<Output = Result<()>> + Send + 'static,
    {
        let desc = description.to_string();
        let child_token = cancel_token.child_token();
        let child_token_clone = child_token.clone();
        let handle = task::spawn(async move {
            match future(state, child_token_clone).await {
                Ok(()) => {
                    log::debug!("{desc} completed successfully");
                    Ok(())
                }
                Err(e) => {
                    log::error!("{desc} failed: {e:?}");
                    cancel_token.cancel();
                    return Err(e);
                }
            }
        });

        Ok(TaskExecutionHandle {
            cancel_token: child_token,
            handle,
        })
    }
}

struct Proxy {
    // this should be a ServiceEndpoint - we need to know the exact subject
    endpoint: Endpoint,
}

impl Proxy {
    fn subject(&self) -> String {
        unimplemented!()
    }

    fn context(&self) -> &async_zmq::Context {
        unimplemented!()
    }
}

// async fn event_loop(proxy: Proxy, cancel_token: CancellationToken) -> Result<()> {
//     let inproc_id = format!("ipc-{}", proxy.subject());
//     let ipc_id = format!("ipc://{}", inproc_id);

//     let socket = proxy.context().socket(zmq::ROUTER)?;
//     socket.set_identity(inproc_id.as_bytes())?;
//     socket.connect(&super::bridge::inproc_endpoint())?;

//     let dealer = Dealer::<IntoIter<Vec<u8>>, Vec<u8>>::from(socket);

//     //dealer.as_raw_socket().set_identity(inproc_id.as_bytes())?;

//     // let (tx, rx) = mpsc::channel(1);

//     // let proxy_handle = TaskExecutionHandle::new(
//     //     proxy,
//     //     move |proxy, cancel_token| async move { Ok(()) },
//     //     cancel_token,
//     //     "zmq endpoint proxy",
//     // )?;
//     Ok(())
// }
