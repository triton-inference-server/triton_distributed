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

use anyhow::Result;
use core::panic;
use std::{collections::HashMap, sync::Arc};
use tokio::sync::Mutex;

use bytes::Bytes;
use derive_builder::Builder;
use futures::StreamExt;
use local_ip_address::{list_afinet_netifas, local_ip};
use serde::{Deserialize, Serialize};
use tokio::{
    io::AsyncWriteExt,
    sync::{mpsc, oneshot},
};
use tokio_util::codec::{FramedRead, FramedWrite};

use super::{
    CallHomeHandshake, PendingConnections, RegisteredStream, StreamOptions, StreamReceiver,
    StreamSender, TcpStreamConnectionInfo, TwoPartCodec,
};
use crate::engine::AsyncEngineContext;
use crate::pipeline::{
    network::{
        codec::{TwoPartMessage, TwoPartMessageType},
        tcp::StreamType,
        ResponseService, ResponseStreamPrologue,
    },
    PipelineError,
};

#[allow(dead_code)]
type ResponseType = TwoPartMessage;

#[derive(Debug, Serialize, Deserialize, Clone, Builder, Default)]
pub struct ServerOptions {
    #[builder(default = "0")]
    pub port: u16,

    #[builder(default)]
    pub interface: Option<String>,
}

impl ServerOptions {
    pub fn builder() -> ServerOptionsBuilder {
        ServerOptionsBuilder::default()
    }
}

// todo - rename TcpResponseServer
// we may need to disambiguate this and a TcpRequestServer

/// A [`TcpStreamServer`] is a TCP service that listens on a port for incoming response connections.
/// A Response connection is a connection that is established by a client with the intention of sending
/// specific data back to the server. The key differentiating factor is that a [`ResponseServer`] is
/// expecting a connection from a client with an established subject.
pub struct TcpStreamServer {
    local_ip: String,
    local_port: u16,
    state: Arc<Mutex<State>>,
}

// pub struct TcpStreamReceiver {
//     address: TcpStreamConnectionInfo,
//     state: Arc<Mutex<State>>,
//     rx: mpsc::Receiver<ResponseType>,
// }

#[allow(dead_code)]
struct RequestedSendConnection {
    context: Arc<dyn AsyncEngineContext>,
    connection: oneshot::Sender<Result<StreamSender, String>>,
}

struct RequestedRecvConnection {
    context: Arc<dyn AsyncEngineContext>,
    connection: oneshot::Sender<Result<StreamReceiver, String>>,
}

// /// When registering a new TcpStream on the server, the registration method will return a [`Connections`] object.
// /// This [`Connections`] object will have two [`oneshot::Receiver`] objects, one for the [`TcpStreamSender`] and one for the [`TcpStreamReceiver`].
// /// The [`Connections`] object can be awaited to get the [`TcpStreamSender`] and [`TcpStreamReceiver`] objects; these objects will
// /// be made available when the matching Client has connected to the server.
// pub struct Connections {
//     pub address: TcpStreamConnectionInfo,

//     /// The [`oneshot::Receiver`] for the [`TcpStreamSender`]. Awaiting this object will return the [`TcpStreamSender`] object once
//     /// the client has connected to the server.
//     pub sender: Option<oneshot::Receiver<StreamSender>>,

//     /// The [`oneshot::Receiver`] for the [`TcpStreamReceiver`]. Awaiting this object will return the [`TcpStreamReceiver`] object once
//     /// the client has connected to the server.
//     pub receiver: Option<oneshot::Receiver<StreamReceiver>>,
// }

#[derive(Default)]
struct State {
    tx_subjects: HashMap<String, RequestedSendConnection>,
    rx_subjects: HashMap<String, RequestedRecvConnection>,
    handle: Option<tokio::task::JoinHandle<()>>,
}

impl TcpStreamServer {
    pub fn options_builder() -> ServerOptionsBuilder {
        ServerOptionsBuilder::default()
    }

    pub async fn new(options: ServerOptions) -> Result<Arc<Self>, PipelineError> {
        let local_ip = match options.interface {
            Some(interface) => {
                let interfaces: HashMap<String, std::net::IpAddr> =
                    list_afinet_netifas()?.into_iter().collect();

                interfaces
                    .get(&interface)
                    .ok_or(PipelineError::Generic(format!(
                        "Interface not found: {}",
                        interface
                    )))?
                    .to_string()
            }
            None => local_ip().unwrap().to_string(),
        };

        let state = Arc::new(Mutex::new(State::default()));

        let local_port = Self::start(local_ip.clone(), options.port, state.clone())
            .await
            .map_err(|e| {
                PipelineError::Generic(format!("Failed to start TcpStreamServer: {}", e))
            })?;

        tracing::info!("TcpStreamServer started on {}:{}", local_ip, local_port);

        Ok(Arc::new(Self {
            local_ip,
            local_port,
            state,
        }))
    }

    #[allow(clippy::await_holding_lock)]
    async fn start(local_ip: String, local_port: u16, state: Arc<Mutex<State>>) -> Result<u16> {
        let addr = format!("{}:{}", local_ip, local_port);
        let state_clone = state.clone();
        let mut guard = state.lock().await;
        if guard.handle.is_some() {
            panic!("TcpStreamServer already started");
        }
        let (ready_tx, ready_rx) = tokio::sync::oneshot::channel::<Result<u16>>();
        let handle = tokio::spawn(tcp_listener(addr, state_clone, ready_tx));
        guard.handle = Some(handle);
        drop(guard);
        let local_port = ready_rx.await??;
        Ok(local_port)
    }
}

// todo - possible rename ResponseService to ResponseServer
#[async_trait::async_trait]
impl ResponseService for TcpStreamServer {
    /// Register a new subject and sender with the response subscriber
    /// Produces an RAII object that will deregister the subject when dropped
    ///
    /// we need to register both data in and data out entries
    /// there might be forward pipeline that want to consume the data out stream
    /// and there might be a response stream that wants to consume the data in stream
    /// on registration, we need to specific if we want data-in, data-out or both
    /// this will map to the type of service that is runniing, i.e. Single or Many In //
    /// Single or Many Out
    ///
    /// todo(ryan) - return a connection object that can be awaited. when successfully connected,
    /// can ask for the sender and receiver
    ///
    /// OR
    ///
    /// we make it into register sender and register receiver, both would return a connection object
    /// and when a connection is established, we'd get the respective sender or receiver
    ///
    /// the registation probably needs to be done in one-go, so we should use a builder object for
    /// requesting a receiver and optional sender
    async fn register(&self, options: StreamOptions) -> PendingConnections {
        // oneshot channels to pass back the sender and receiver objects

        let address = format!("{}:{}", self.local_ip, self.local_port);
        tracing::debug!("Registering new TcpStream on {}", address);

        let send_stream = if options.enable_request_stream {
            let sender_subject = uuid::Uuid::new_v4().to_string();

            let (pending_sender_tx, pending_sender_rx) = oneshot::channel();

            let connection_info = RequestedSendConnection {
                context: options.context.clone(),
                connection: pending_sender_tx,
            };

            let mut state = self.state.lock().await;
            state
                .tx_subjects
                .insert(sender_subject.clone(), connection_info);

            let registered_stream = RegisteredStream {
                connection_info: TcpStreamConnectionInfo {
                    address: address.clone(),
                    subject: sender_subject.clone(),
                    context: options.context.id().to_string(),
                    stream_type: StreamType::Request,
                }
                .into(),
                stream_provider: pending_sender_rx,
            };

            Some(registered_stream)
        } else {
            None
        };

        let recv_stream = if options.enable_response_stream {
            let (pending_recver_tx, pending_recver_rx) = oneshot::channel();
            let receiver_subject = uuid::Uuid::new_v4().to_string();

            let connection_info = RequestedRecvConnection {
                context: options.context.clone(),
                connection: pending_recver_tx,
            };

            let mut state = self.state.lock().await;
            state
                .rx_subjects
                .insert(receiver_subject.clone(), connection_info);

            let registered_stream = RegisteredStream {
                connection_info: TcpStreamConnectionInfo {
                    address: address.clone(),
                    subject: receiver_subject.clone(),
                    context: options.context.id().to_string(),
                    stream_type: StreamType::Response,
                }
                .into(),
                stream_provider: pending_recver_rx,
            };

            Some(registered_stream)
        } else {
            None
        };

        PendingConnections {
            send_stream,
            recv_stream,
        }
    }
}

// this method listens on a tcp port for incoming connections
// new connections are expected to send a protocol specific handshake
// for us to determine the subject they are interested in, in this case,
// we expect the first message to be [`FirstMessage`] from which we find
// the sender, then we spawn a task to forward all bytes from the tcp stream
// to the sender
async fn tcp_listener(
    addr: String,
    state: Arc<Mutex<State>>,
    read_tx: tokio::sync::oneshot::Sender<Result<u16>>,
) {
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to start TcpListender on {}: {}", addr, e));

    let listener = match listener {
        Ok(listener) => {
            let addr = listener
                .local_addr()
                .map_err(|e| anyhow::anyhow!("Failed get SocketAddr: {:?}", e))
                .unwrap();

            read_tx
                .send(Ok(addr.port()))
                .expect("Failed to send ready signal");

            listener
        }
        Err(e) => {
            read_tx.send(Err(e)).expect("Failed to send ready signal");
            return;
        }
    };

    loop {
        let (stream, _addr) = listener.accept().await.unwrap();
        stream.set_nodelay(true).unwrap();
        tokio::spawn(handle_connection(stream, state.clone()));
    }

    // #[instrument(level = "trace"), skip(state)]
    // todo - clone before spawn and trace process_stream
    async fn handle_connection(stream: tokio::net::TcpStream, state: Arc<Mutex<State>>) {
        let result = process_stream(stream, state).await;
        match result {
            Ok(_) => tracing::trace!("TcpStream connection closed"),
            Err(e) => tracing::error!("TcpStream connection failed: {}", e),
        }
    }

    /// This method is responsible for the internal tcp stream handshake
    /// The handshake will specialize the stream as a request/sender or response/receiver stream
    async fn process_stream(
        stream: tokio::net::TcpStream,
        state: Arc<Mutex<State>>,
    ) -> Result<(), String> {
        // split the socket in to a reader and writer
        let (read_half, write_half) = tokio::io::split(stream);

        // attach the codec to the reader and writer to get framed readers and writers
        let mut framed_reader = FramedRead::new(read_half, TwoPartCodec::default());
        let framed_writer = FramedWrite::new(write_half, TwoPartCodec::default());

        // the internal tcp [`CallHomeHandshake`] connects the socket to the requester
        // here we await this first message as a raw bytes two part message
        let first_message = framed_reader
            .next()
            .await
            .ok_or("Connection closed without a ControlMessge".to_string())?
            .map_err(|e| e.to_string())?;

        // we await on the raw bytes which should come in as a header only message
        // todo - improve error handling - check for no data
        if first_message.header().is_none() {
            return Err("Expected ControlMessage, got DataMessage".to_string());
        }

        // deserialize the [`CallHomeHandshake`] message
        let handshake: CallHomeHandshake = serde_json::from_slice(first_message.header().unwrap())
            .map_err(|e| {
                format!(
                    "Failed to deserialize the first message as a valid `CallHomeHandshake`: {}",
                    e
                )
            })?;

        // branch here to handle sender stream or receiver stream
        match handshake.stream_type {
            StreamType::Request => process_request_stream().await,
            StreamType::Response => {
                process_response_stream(handshake.subject, state, framed_reader, framed_writer)
                    .await
            }
        }
        .map_err(|e| format!("Failed to process stream: {}", e))
    }

    async fn process_request_stream() -> Result<(), String> {
        Ok(())
    }

    async fn process_response_stream(
        subject: String,
        state: Arc<Mutex<State>>,
        mut reader: FramedRead<tokio::io::ReadHalf<tokio::net::TcpStream>, TwoPartCodec>,
        writer: FramedWrite<tokio::io::WriteHalf<tokio::net::TcpStream>, TwoPartCodec>,
    ) -> Result<(), String> {
        let response_stream = state
            .lock().await
            .rx_subjects
            .remove(&subject)
            .ok_or(format!("Subject not found: {}; upstream publisher specified a subject unknown to the downsteam subscriber", subject))?;

        // unwrap response_stream
        let RequestedRecvConnection {
            context,
            connection,
        } = response_stream;

        // the [`Prologue`]
        // there must be a second control message it indicate the other segment's generate method was successful
        let prologue = reader
            .next()
            .await
            .ok_or("Connection closed without a ControlMessge".to_string())?
            .map_err(|e| e.to_string())?;

        // deserialize prologue
        let prologue = match prologue.into_message_type() {
            TwoPartMessageType::HeaderOnly(header) => {
                let prologue: ResponseStreamPrologue = serde_json::from_slice(&header)
                    .map_err(|e| format!("Failed to deserialize ControlMessage: {}", e))?;
                prologue
            }
            _ => {
                panic!("Expected HeaderOnly ControlMessage; internally logic error")
            }
        };

        // await the control message of GTG or Error, if error, then connection.send(Err(String)), which should fail the
        // generate call chain
        //
        // note: this second control message might be delayed, but the expensive part of setting up the connection
        // is both complete and ready for data flow; awaiting here is not a performance hit or problem and it allows
        // us to trace the initial setup time vs the time to prologue
        if let Some(error) = &prologue.error {
            let _ = connection.send(Err(error.clone()));
            return Err(format!("Received error prologue: {}", error));
        }

        // we need to know the buffer size from the registration options; add this to the RequestRecvConnection object
        let (tx, rx) = mpsc::channel(16);

        if connection
            .send(Ok(crate::pipeline::network::StreamReceiver { rx }))
            .is_err()
        {
            return Err("The requester of the stream has been dropped before the connection was established".to_string());
        }

        let (alive_tx, alive_rx) = mpsc::channel::<()>(1);
        let (control_tx, _control_rx) = mpsc::channel::<Bytes>(8);

        // monitor task
        // if the context is cancelled, we need to forward the message across the transport layer
        // we only termine the forwarding task on a kill signal, on a stop signal, we issue the stop signal, then await for the producer
        // to naturally close the stream
        let monitor_task = tokio::spawn(monitor(writer, context.clone(), alive_tx));

        // forward task
        let forward_task = tokio::spawn(handle_response_stream(
            reader,
            tx,
            control_tx,
            context.clone(),
            alive_rx,
        ));

        // check the results of each of the tasks
        let (monitor_result, forward_result) = tokio::join!(monitor_task, forward_task);

        // if either of the tasks failed, we need to return an error
        if let Err(e) = monitor_result {
            return Err(format!("Monitor task failed: {}", e));
        }
        if let Err(e) = forward_result {
            return Err(format!("Forward task failed: {}", e));
        }

        Ok(())
    }

    async fn handle_response_stream(
        mut framed_reader: FramedRead<tokio::io::ReadHalf<tokio::net::TcpStream>, TwoPartCodec>,
        response_tx: mpsc::Sender<Bytes>,
        control_tx: mpsc::Sender<Bytes>,
        context: Arc<dyn AsyncEngineContext>,
        alive_rx: mpsc::Receiver<()>,
    ) -> Result<(), String> {
        // loop over reading the tcp stream and checking if the writer is closed
        loop {
            tokio::select! {
                msg = framed_reader.next() => {
                    match msg {
                        Some(Ok(msg)) => {
                            let (header, data) = msg.into_parts();

                            if !header.is_empty() && (control_tx.send(header).await).is_err() {
                                tracing::trace!("Control channel closed")
                            }

                            if !data.is_empty() {
                                response_tx.send(data).await.unwrap();
                            }
                        }
                        Some(Err(e)) => {
                            return Err(format!("Failed to read TwoPartCodec message from TcpStream: {}", e));
                        }
                        None => {
                            tracing::trace!("TcpStream closed naturally");
                            break;
                        }
                    }
                }
                _ = response_tx.closed() => {
                    break;
                }
                _ = context.killed() => { break; }
            }
        }
        drop(alive_rx);
        Ok(())
    }

    #[allow(dead_code)]
    async fn handle_control_message(
        mut control_rx: mpsc::Receiver<Bytes>,
        context: Arc<dyn AsyncEngineContext>,
        alive_tx: mpsc::Sender<()>,
    ) -> Result<(), String> {
        loop {
            tokio::select! {
                msg = control_rx.recv() => {
                    match msg {
                        Some(_msg) => {
                            // handle control message
                        }
                        None => {
                            tracing::trace!("Control channel closed");
                            break;
                        }
                    }
                }
                _ = context.killed() => {
                    break;
                }
            }
        }
        drop(alive_tx);
        Ok(())
    }

    async fn monitor(
        _socket_tx: FramedWrite<tokio::io::WriteHalf<tokio::net::TcpStream>, TwoPartCodec>,
        ctx: Arc<dyn AsyncEngineContext>,
        alive_tx: mpsc::Sender<()>,
    ) {
        let alive_tx = alive_tx;
        tokio::select! {
            _ = ctx.stopped() => {
                // send cancellation message
                panic!("impl cancellation signal");
            }
            _ = alive_tx.closed() => {
                tracing::trace!("response stream closed naturally")
            }
        }
        let mut framed_writer = _socket_tx;
        framed_writer.get_mut().shutdown().await.unwrap();
    }
}
