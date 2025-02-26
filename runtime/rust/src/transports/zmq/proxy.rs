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

use std::{future::Future, sync::Arc, vec::IntoIter};

use async_trait::async_trait;
use async_zmq::{zmq, Dealer};
use tokio::{
    sync::{mpsc, oneshot},
    task,
};
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::sync::CancellationToken;

use derive_builder::Builder;

use crate::{
    component::Endpoint,
    engine::{AsyncEngine, AsyncEngineContextProvider, ResponseStream},
    pipeline::{ManyOut, SingleIn},
    raise, Error, Result,
};

pub struct ZmqProxy {}

impl ZmqProxy {
    pub fn new(endpoint: Endpoint) -> Result<Arc<Self>> {
        let proxy = Arc::new(Self {});
        Ok(proxy)
    }

    async fn enqueue(
        &self,
        request: Vec<u8>,
    ) -> Result<oneshot::Receiver<mpsc::Receiver<Vec<u8>>>> {
        todo!()
    }
}

#[async_trait]
impl AsyncEngine<SingleIn<Vec<u8>>, ManyOut<Vec<u8>>, Error> for ZmqProxy {
    async fn generate(&self, request: SingleIn<Vec<u8>>) -> Result<ManyOut<Vec<u8>>> {
        let (request, context) = request.into_parts();
        let pending_rx = self.enqueue(request).await?;
        let rx = pending_rx.await?;
        let stream = ReceiverStream::new(rx);
        Ok(ResponseStream::new(Box::pin(stream), context.context()))
    }
}

enum ProxyOperation {
    Enqueue(Vec<u8>),
    Cancel(uuid::Uuid),
    Shutdown,
}

struct ProxyState {
    endpoint: Endpoint,
    context: async_zmq::Context,
    ops_rx: mpsc::Receiver<ProxyOperation>,
}

async fn event_loop(state: ProxyState, cancel_token: CancellationToken) -> Result<()> {
    // let subject = state.endpoint.subject();
    // let inproc_id = format!("ipc-{}", subject);
    // let ipc_id = format!("ipc://{}", subject);

    // let socket = state.context.socket(zmq::ROUTER)?;
    // socket.set_identity(inproc_id.as_bytes())?;
    // socket.connect(&super::bridge::inproc_endpoint())?;

    // let dealer = Dealer::<IntoIter<Vec<u8>>, Vec<u8>>::from(socket);

    //dealer.as_raw_socket().set_identity(inproc_id.as_bytes())?;

    // let (tx, rx) = mpsc::channel(1);

    // let proxy_handle = TaskExecutionHandle::new(
    //     proxy,
    //     move |proxy, cancel_token| async move { Ok(()) },
    //     cancel_token,
    //     "zmq endpoint proxy",
    // )?;
    Ok(())
}
