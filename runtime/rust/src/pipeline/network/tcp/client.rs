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

use std::sync::Arc;

use futures::{SinkExt, StreamExt};
use tokio::io::{ReadHalf, WriteHalf};
use tokio::{io::AsyncWriteExt, net::TcpStream};
use tokio_util::codec::{FramedRead, FramedWrite};

use super::{CallHomeHandshake, ControlMessage, TcpStreamConnectionInfo};
use crate::engine::AsyncEngineContext;
use crate::pipeline::network::{
    codec::{TwoPartCodec, TwoPartMessage},
    tcp::StreamType,
    ConnectionInfo, ResponseStreamPrologue, StreamSender,
}; // Import SinkExt to use the `send` method

#[allow(dead_code)]
pub struct TcpClient {
    worker_id: String,
}

impl Default for TcpClient {
    fn default() -> Self {
        TcpClient {
            worker_id: uuid::Uuid::new_v4().to_string(),
        }
    }
}

impl TcpClient {
    pub fn new(worker_id: String) -> Self {
        TcpClient { worker_id }
    }

    async fn connect(address: &str) -> Result<TcpStream, String> {
        let socket = TcpStream::connect(address)
            .await
            .map_err(|e| format!("failed to connect: {:?}", e))?;

        socket
            .set_nodelay(true)
            .map_err(|e| format!("failed to set nodelay: {:?}", e))?;

        Ok(socket)
    }

    pub async fn create_response_steam(
        context: Arc<dyn AsyncEngineContext>,
        info: ConnectionInfo,
    ) -> Result<StreamSender, String> {
        let info = TcpStreamConnectionInfo::try_from(info)?;
        tracing::trace!("Creating response stream for {:?}", info);

        if info.stream_type != StreamType::Response {
            return Err(format!(
                "Invalid stream type; TcpClient requires the stream type to be `response`; however {:?} was passed",
                info.stream_type
            ));
        }

        if info.context != context.id() {
            return Err(format!(
                "Invalid context; TcpClient requires the context to be {:?}; however {:?} was passed",
                context.id(),
                info.context
            ));
        }

        let stream = TcpClient::connect(&info.address).await?;
        let (read_half, write_half) = tokio::io::split(stream);

        let framed_reader = FramedRead::new(read_half, TwoPartCodec::default());
        let mut framed_writer = FramedWrite::new(write_half, TwoPartCodec::default());

        // this is a oneshot channel that will be used to signal when the stream is closed
        // when the stream sender is dropped, the bytes_rx will be closed and the forwarder task will exit
        // the forwarder task will capture the alive_rx half of the oneshot channel; this will close the alive channel
        // so the holder of the alive_tx half will be notified that the stream is closed; the alive_tx channel will be
        // captured by the monitor task
        let (alive_tx, alive_rx) = tokio::sync::oneshot::channel::<()>();

        tokio::spawn(control_handler(framed_reader, context, alive_tx));

        // transport specific handshake message
        let handshake = CallHomeHandshake {
            subject: info.subject,
            stream_type: StreamType::Response,
        };

        let handshake_bytes = match serde_json::to_vec(&handshake) {
            Ok(hb) => hb,
            Err(err) => {
                return Err(format!(
                    "create_response_steam: Error converting CallHomeHandshake to JSON array: {err:#}"
                ));
            }
        };
        let msg = TwoPartMessage::from_header(handshake_bytes.into());

        // issue the the first tcp handshake message
        framed_writer
            .send(msg)
            .await
            .map_err(|e| format!("failed to send handshake: {:?}", e))?;

        // set up the channel to send bytes to the transport layer
        let (bytes_tx, bytes_rx) = tokio::sync::mpsc::channel(16);

        // forwards the bytes send from this stream to the transport layer; hold the alive_rx half of the oneshot channel
        tokio::spawn(forward_handler(bytes_rx, framed_writer, alive_rx));

        // set up the prologue for the stream
        // this might have transport specific metadata in the future
        let prologue = Some(ResponseStreamPrologue { error: None });

        // create the stream sender
        let stream_sender = StreamSender {
            tx: bytes_tx,
            prologue,
        };

        Ok(stream_sender)
    }
}

/// monitors the channel for a cancellation signal
/// this task exits when the alive_rx half of the oneshot channel is closed or a stop/kill signal is received
async fn control_handler(
    mut framed_reader: FramedRead<ReadHalf<TcpStream>, TwoPartCodec>,
    context: Arc<dyn AsyncEngineContext>,
    mut alive_tx: tokio::sync::oneshot::Sender<()>,
) {
    loop {
        tokio::select! {
            msg = framed_reader.next() => {
                match msg {
                    Some(Ok(two_part_msg)) => {
                        match two_part_msg.optional_parts() {
                           (Some(bytes), None) => {
                                let msg: ControlMessage = match serde_json::from_slice(bytes) {
                                    Ok(msg) => msg,
                                    Err(err) => {
                                        let json_str = String::from_utf8_lossy(bytes);
                                        tracing::error!(%err, %json_str, "control_handler fatal error deserializing JSON to ControlMessage");
                                        context.kill();
                                        break;
                                    }
                                };
                                match msg {
                                    ControlMessage::Stop => {
                                        context.stop();
                                        break;
                                    }
                                    ControlMessage::Kill => {
                                        context.kill();
                                        break;
                                    }
                                }
                           }
                           _ => {
                               // we should not receive this
                           }
                        }
                    }
                    Some(Err(e)) => {
                        panic!("failed to decode message from stream: {:?}", e);
                        // break;
                    }
                    None => {
                        // the stream was closed, we should stop the stream
                        return;
                    }
                }
            }
            _ = alive_tx.closed() => {
                // the channel was closed, we should stop the stream
                break;
            }
        }
    }
    // framed_writer.get_mut().shutdown().await.unwrap();
}

async fn forward_handler(
    mut bytes_rx: tokio::sync::mpsc::Receiver<TwoPartMessage>,
    mut framed_writer: FramedWrite<WriteHalf<TcpStream>, TwoPartCodec>,
    alive_rx: tokio::sync::oneshot::Receiver<()>,
) {
    while let Some(msg) = bytes_rx.recv().await {
        if let Err(e) = framed_writer.send(msg).await {
            tracing::trace!(%e, "failed to send message to stream; possible disconnect");

            // TODO - possibly propagate the error upstream
            break;
        }
    }
    drop(alive_rx);
    if let Err(e) = framed_writer.get_mut().shutdown().await {
        tracing::trace!("failed to shutdown writer: {:?}", e);
    }
}
