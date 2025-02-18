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

//! ZMQ Transport
//!
//! This module provides a ZMQ transport for the [crate::DistributedRuntime].
//!
//! Currently, the [Server] consists of a [async_zmq::Router] and the [Client] leverages
//! a [async_zmq::Dealer].
//!
//! The distributed service pattern we will use is based on the Harmony pattern described in
//! [Chapter 8: A Framework for Distributed Computing](https://zguide.zeromq.org/docs/chapter8/#True-Peer-Connectivity-Harmony-Pattern).
//!
//! This is similar to the TCP implementation; however, the TCP implementation used a direct
//! connection between the client and server per stream. The ZMQ transport will enable the
//! equivalent of a connection pool per upstream service at the cost of needing an extra internal
//! routing step per service endpoint.

use async_trait::async_trait;
use async_zmq::{Context, Dealer, Message, Router, Sink, SinkExt, StreamExt};
use bytes::Bytes;
use derive_getters::Dissolve;
use futures::TryStreamExt;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, os::fd::FromRawFd, sync::Arc, time::Duration, vec::IntoIter};
use tokio::{
    sync::{mpsc, oneshot, Mutex},
    task::{JoinError, JoinHandle},
};
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::sync::CancellationToken;
use tracing as log;

use crate::{
    engine::{AsyncEngine, AsyncEngineContextProvider, ResponseStream},
    error,
    pipeline::{ManyOut, SingleIn},
    raise, Error, ErrorContext, Result,
};

pub mod bridge;
pub mod endpoint;
pub mod proxy;

type Frame = Vec<Message>;

// Received message from the Router have the following frame layout:

/// Expected message count for messages received from the ZMQ Router
pub const ZMQ_ROUTER_EXPECTED_MESSAGE_COUNT: usize = 3;

/// Identity index in the message
pub const ZMQ_ROUTER_IDENTITY_INDEX: usize = 0;

/// Request ID index in the message
pub const ZMQ_ROUTER_REQUEST_ID_INDEX: usize = 1;

/// Message type index in the message
pub const ZMQ_ROUTER_MESSAGE_PAYLOAD_INDEX: usize = 2;

// Core message types
#[derive(Debug, Clone, Serialize, Deserialize)]
enum ControlMessage {
    Cancel { request_id: String },
    CancelAck { request_id: String },
    Error { request_id: String, error: String },
    Complete { request_id: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum MessageType {
    Data(Vec<u8>),
    Control(ControlMessage),
}

enum StreamAction {
    SendEager(usize),
    SendDelayed(usize),
    Close,
}

// Router state management
struct RouterState {
    active_streams: HashMap<String, mpsc::Sender<Frame>>,
    control_channels: HashMap<String, mpsc::Sender<ControlMessage>>,
}

impl RouterState {
    fn new() -> Self {
        Self {
            active_streams: HashMap::new(),
            control_channels: HashMap::new(),
        }
    }

    fn register_stream(
        &mut self,
        request_id: String,
        data_tx: mpsc::Sender<Frame>,
        control_tx: mpsc::Sender<ControlMessage>,
    ) {
        self.active_streams.insert(request_id.clone(), data_tx);
        self.control_channels.insert(request_id, control_tx);
    }

    fn remove_stream(&mut self, request_id: &str) {
        self.active_streams.remove(request_id);
        self.control_channels.remove(request_id);
    }
}

#[derive(Dissolve)]
struct ZmqRouterSend {
    /// Identity of the dealer
    target_id: Vec<u8>,

    /// Message Stream ID
    stream_id: String,

    /// User request
    message: Vec<u8>,

    /// Response channel
    tx: oneshot::Sender<Result<mpsc::Receiver<Frame>>>,
}

#[derive(Dissolve)]
struct ZmqRouterRegister {
    /// Message Stream ID
    stream_id: String,

    /// Response channel
    tx: oneshot::Sender<Result<mpsc::Receiver<Frame>>>,
}

enum ZmqRouterOperation {
    /// Send a message to a stream
    Send(ZmqRouterSend),

    /// Register a stream_id to receive messages
    Register(ZmqRouterRegister),

    /// Drop a stream
    Drop(String),
}

// Server implementation
#[derive(Clone, Dissolve)]
pub struct Server {
    cancel_token: CancellationToken,
    req_tx: mpsc::Sender<ZmqRouterOperation>,
    fd: i32,
}

impl Server {
    /// Create a new [Server] which is a [async_zmq::Router] with the given [async_zmq::Context] and address to bind
    /// the ZMQ [async_zmq::Router] socket.
    ///
    /// If the event loop processing the router fails with an error, the signal is propagated through the [CancellationToken]
    /// by issuing a [CancellationToken::cancel].
    ///
    /// The [Server] is how you interact with the running instance.
    ///
    /// The [ServerExecutionHandle] is the handle for background task executing the [Server].
    pub async fn new(
        context: &Context,
        address: &str,
        cancel_token: CancellationToken,
    ) -> Result<(Self, ServerExecutionHandle)> {
        let router = async_zmq::router(address)?.with_context(context).bind()?;
        let fd = router.as_raw_socket().get_fd()?;

        // send channel
        let (tx, rx) = mpsc::channel(1024);

        // cancel the router's event loop
        let child = cancel_token.child_token();

        // primary event handling task
        // sends addressed messages through the router
        // receives messages from the router, routes them via the state
        let primary_task = tokio::spawn(Self::run(router, child.child_token(), rx));

        // this task captures the primary cancellation token, so if an error occurs, we can cancel the router's event loop
        // but we also propagate the error to the caller's cancellation token
        let watch_task = tokio::spawn(async move {
            let result = primary_task.await.inspect_err(|e| {
                log::error!("zmq server/router task failed: {}", e);
                cancel_token.cancel();
            })?;
            result.inspect_err(|e| {
                log::error!("zmq server/router task failed: {}", e);
                cancel_token.cancel();
            })
        });

        let handle = ServerExecutionHandle {
            task: watch_task,
            cancel_token: child.clone(),
        };

        Ok((
            Self {
                cancel_token: child,
                req_tx: tx,
                fd,
            },
            handle,
        ))
    }

    async fn send_to(
        &self,
        target_id: String,
        stream_id: String,
        message: Vec<u8>,
    ) -> Result<oneshot::Receiver<Result<mpsc::Receiver<Frame>>>> {
        let (tx, rx) = oneshot::channel();

        let request = ZmqRouterSend {
            target_id: target_id.as_bytes().to_vec(),
            stream_id,
            message,
            tx,
        };

        let _ = self
            .req_tx
            .send(ZmqRouterOperation::Send(request))
            .await
            .context("Failed to send request / engine is shutting down")?;

        Ok(rx)
    }

    // pub async fn register_stream(&)

    async fn run(
        router: Router<IntoIter<Vec<u8>>, Vec<u8>>,
        token: CancellationToken,
        op_rx: mpsc::Receiver<ZmqRouterOperation>,
    ) -> Result<()> {
        let mut router = router;
        let mut op_rx = op_rx;
        let mut state = RouterState::new();

        // todo - move this into the Server impl to discover the os port being used
        // let fd = router.as_raw_socket().get_fd()?;
        // let sock = unsafe { socket2::Socket::from_raw_fd(fd) };
        // let addr = sock.local_addr()?;
        // let port = addr.as_socket().map(|s| s.port());

        // if let Some(port) = port {
        //     log::info!("Server listening on port {}", port);
        // }

        loop {
            let frames = tokio::select! {

                // if the request channel is closed, we are done writing
                // todo: determine policy if should continue or exit
                Some(op) = op_rx.recv(), if !op_rx.is_closed() => {
                    handle_operation(&mut router, &mut state, op).await;
                    continue;
                }

                frames = router.next() => {
                    match frames {
                        Some(Ok(frames)) => {
                            frames
                        },
                        Some(Err(e)) => {
                            log::warn!("Error receiving message: {}", e);
                            continue;
                        }
                        None => break,
                    }
                }

                _ = token.cancelled() => {
                    log::info!("Server shutting down");
                    break;
                }
            };

            // we should have at least 3 frames
            // 0: identity
            // 1: request_id
            // 2: message type

            // if the contract is broken, we should exit
            if frames.len() != ZMQ_ROUTER_EXPECTED_MESSAGE_COUNT {
                panic!(
                    "Fatal Error -- Broken contract -- Expected 3 frames, got {}",
                    frames.len()
                );
            }

            let bytes = frames.iter().map(|f| f.len()).sum();

            let request_id =
                String::from_utf8_lossy(&frames[ZMQ_ROUTER_REQUEST_ID_INDEX]).to_string();

            if let Some(tx) = state.active_streams.get(&request_id) {
                // first we try to send the data eagerly without blocking
                let action = match tx.try_send(frames) {
                    Ok(_) => {
                        log::trace!(
                            request_id,
                            "response data sent eagerly to stream: {} bytes",
                            bytes
                        );
                        StreamAction::SendEager(bytes)
                    }
                    Err(e) => match e {
                        mpsc::error::TrySendError::Closed(_) => {
                            log::info!(request_id, "response stream was closed");
                            StreamAction::Close
                        }
                        mpsc::error::TrySendError::Full(data) => {
                            log::warn!(request_id, "response stream is full; backpressue alert");
                            // todo - add timeout - we are blocking all other streams
                            if (tx.send(data).await).is_err() {
                                StreamAction::Close
                            } else {
                                StreamAction::SendDelayed(bytes)
                            }
                        }
                    },
                };

                match action {
                    StreamAction::SendEager(_size) => {
                        // increment bytes_received
                        // increment messages_received
                        // increment eager_messages_received
                    }
                    StreamAction::SendDelayed(_size) => {
                        // increment bytes_received
                        // increment messages_received
                        // increment delayed_messages_received
                    }
                    StreamAction::Close => {
                        state.active_streams.remove(&request_id);
                    }
                }
            } else {
                // increment bytes_dropped
                // increment messages_dropped
                log::trace!(request_id, "no active stream for request_id");
            }
        }

        Ok(())
    }
}

async fn handle_operation(
    router: &mut Router<IntoIter<Vec<u8>>, Vec<u8>>,
    state: &mut RouterState,
    operation: ZmqRouterOperation,
) {
    match operation {
        ZmqRouterOperation::Send(send) => {
            let (target_id, stream_id, message, tx) = send.dissolve();
            match handle_send(router, state, target_id, &stream_id, message).await {
                Ok(rx) => {
                    if tx.send(Ok(rx)).is_err() {
                        log::debug!("downstream stream closed");
                        state.active_streams.remove(&stream_id);
                    }
                }
                Err(e) => {
                    if tx.send(Err(e)).is_err() {
                        // nothing when right...
                        log::debug!("downstream stream closed; and failed to send error");
                    }
                }
            }
        }
        ZmqRouterOperation::Register(register) => {
            let (stream_id, tx) = register.dissolve();
            match handle_register(state, &stream_id) {
                Ok(rx) => {
                    if tx.send(Ok(rx)).is_err() {
                        log::debug!("downstream stream closed");
                        state.active_streams.remove(&stream_id);
                    }
                }
                Err(e) => {
                    if tx.send(Err(e)).is_err() {
                        log::debug!("downstream stream closed; and failed to send error");
                    }
                }
            }
        }
        ZmqRouterOperation::Drop(request_id) => {
            state.active_streams.remove(&request_id);
        }
    }
}

async fn handle_send(
    router: &mut Router<IntoIter<Vec<u8>>, Vec<u8>>,
    state: &mut RouterState,
    target_id: Vec<u8>,
    stream_id: &String,
    message: Vec<u8>,
) -> Result<mpsc::Receiver<Frame>> {
    let (tx, rx) = mpsc::channel(1024);
    if state.active_streams.contains_key(stream_id) {
        raise!("stream already exists");
    }

    let _ = router
        .send(
            vec![
                target_id,
                b"".to_vec(),
                stream_id.as_bytes().to_vec(),
                message,
            ]
            .into(),
        )
        .await?;

    state.active_streams.insert(stream_id.to_string(), tx);

    Ok(rx)
}

fn handle_register(state: &mut RouterState, stream_id: &String) -> Result<mpsc::Receiver<Frame>> {
    if state.active_streams.contains_key(stream_id) {
        raise!("stream already exists");
    }

    let (tx, rx) = mpsc::channel(1024);
    state.active_streams.insert(stream_id.to_string(), tx);

    Ok(rx)
}

/// The [ZmqRouterAsyncEngine] is the engine for the [Server]. When a [ZmqRouterAsyncEngine::generate]
/// is called, the request will be sent over ZMQ to the `identity` which is responsible for handling
/// the request and returning a stream of responses termining in a Sentinel message.
pub struct ZmqRouterAsyncEngine {
    identity: String,
    router: Server,
}

impl ZmqRouterAsyncEngine {}

// todo - use Vec<u8> instead of Bytes
#[async_trait]
impl AsyncEngine<SingleIn<Vec<u8>>, ManyOut<Vec<u8>>, Error> for ZmqRouterAsyncEngine {
    async fn generate(&self, request: SingleIn<Vec<u8>>) -> Result<ManyOut<Vec<u8>>, Error> {
        let (request, context) = request.into_parts();

        // we are going to want to get a control message first to indicate if it was a success
        // this will allow us to pack the identity of the sender into the context

        let mut rx = self
            .router
            .send_to(
                self.identity.clone(),
                context.id().to_string(),
                request.to_vec(),
            )
            .await?
            .await??;

        let stream = async_stream::stream! {
            while let Some(frame) = rx.recv().await {
                match frame.get(ZMQ_ROUTER_MESSAGE_PAYLOAD_INDEX) {
                    Some(payload) => yield payload.to_vec(),

                    // this is a fatal error condition, this should never happen
                    None => assert_eq!(ZMQ_ROUTER_EXPECTED_MESSAGE_COUNT, frame.len(),
                        "Fatal Error -- Broken contract -- Expected {ZMQ_ROUTER_EXPECTED_MESSAGE_COUNT} frames, got {}", frame.len()),
                }
            }
        };

        let stream = ResponseStream::new(Box::pin(stream), context.context());
        Ok(stream)
    }
}

struct ZmqIngress {}

/// The [ServerExecutionHandle] is the handle for background task executing the [Server].
///
/// You can use this to check if the server is finished or cancelled.
///
/// You can also join on the task to wait for it to finish.
pub struct ServerExecutionHandle {
    task: JoinHandle<Result<()>>,
    cancel_token: CancellationToken,
}

impl ServerExecutionHandle {
    /// Check if the task awaiting on the [Server]s background event loop has finished.
    pub fn is_finished(&self) -> bool {
        self.task.is_finished()
    }

    /// Check if the server's event loop has been cancelled.
    pub fn is_cancelled(&self) -> bool {
        self.cancel_token.is_cancelled()
    }

    /// Cancel the server's event loop.
    ///
    /// This will signal the server to stop processing requests and exit.
    ///
    /// This will not wait for the server to finish, it will exit immediately.
    ///
    /// This will not propagate to the [CancellationToken] used to start the [Server]
    /// unless an error happens during the shutdown process.
    pub fn cancel(&self) {
        self.cancel_token.cancel();
    }

    /// Join on the task awaiting on the [Server]s background event loop.
    ///
    /// This will return the result of the [Server]s background event loop.
    pub async fn join(self) -> Result<()> {
        self.task.await?
    }
}

// Client implementation
struct Client {
    dealer: Dealer<IntoIter<Vec<u8>>, Vec<u8>>,
}

impl Client {
    fn new(context: &Context, address: &str) -> Result<Self> {
        let dealer = async_zmq::dealer(address)?
            .with_context(context)
            .connect()?;

        Ok(Self { dealer })
    }

    fn dealer(&mut self) -> &mut Dealer<IntoIter<Vec<u8>>, Vec<u8>> {
        &mut self.dealer
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_zmq::zmq;
    use tokio::time::timeout;

    #[tokio::test]
    async fn test_router_bridge_with_dealers() -> Result<()> {
        let context = Context::new();
        let token = CancellationToken::new();
        let lease_id = 1337;

        let handle = bridge::run_router_bridge(context.clone(), lease_id, token.clone())?;

        let foo_id = "inproc-foo";
        let bar_id = "ipc-bar";

        let foo = context.socket(zmq::DEALER)?;
        foo.set_identity(foo_id.as_bytes())?;
        foo.connect(&bridge::inproc_endpoint())?;
        let mut foo = Dealer::<IntoIter<Vec<u8>>, Vec<u8>>::from(foo);

        let bar = context.socket(zmq::DEALER)?;
        bar.set_identity(bar_id.as_bytes())?;
        bar.connect(&bridge::ipc_endpoint(lease_id))?;
        let mut bar = Dealer::<IntoIter<Vec<u8>>, Vec<u8>>::from(bar);

        // multipart format:
        // let remote = &msg[0]; // b"" if local
        // let target = &msg[1]; // identity of dealer
        // let action = &msg[2]; // action to take on dealer
        // let stream = &msg[3]; // stream_id of task on dealer
        // let payload = &msg[4]; // payload
        println!("sending message");
        foo.send(
            vec![
                b"".to_vec(),
                bar_id.as_bytes().to_vec(),
                b"".to_vec(),
                b"".to_vec(),
                b"hi".to_vec(),
            ]
            .into(),
        )
        .await?;

        println!("waiting for message");
        let message = bar.next().await.unwrap().unwrap();
        println!("message: {:?}", message);
        assert_eq!(message.len(), 5);
        println!("message[1]: {:?}", message[1].as_str());
        assert_eq!(message[4].as_str().unwrap(), "hi");

        bar.send(
            vec![
                b"".to_vec(),
                message[1].to_vec(),
                b"".to_vec(),
                b"".to_vec(),
                b"world".to_vec(),
            ]
            .into(),
        )
        .await?;

        let message = foo.next().await.unwrap().unwrap();
        println!("message: {:?}", message);
        assert_eq!(message.len(), 5);
        println!("message[1]: {:?}", message[1].as_str());
        assert_eq!(message[4].as_str().unwrap(), "world");

        token.cancel();
        handle.handle.join().unwrap();
        println!("done");

        Ok(())
    }
}
