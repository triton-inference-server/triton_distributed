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
};
use crate::{error, ErrorContext, Result}; // Import SinkExt to use the `send` method

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

    async fn connect(address: &str) -> std::io::Result<TcpStream> {
        // try to connect to the address; retry with exponential backoff if AddrNotAvailable
        let backoff = std::time::Duration::from_millis(200);
        loop {
            match TcpStream::connect(address).await {
                Ok(socket) => {
                    socket.set_nodelay(true)?;
                    return Ok(socket);
                }
                Err(e) => {
                    if e.kind() == std::io::ErrorKind::AddrNotAvailable {
                        tracing::warn!("retry warning: failed to connect: {:?}", e);
                        eprintln!("retry warning: failed to connect: {:?}", e);
                        tokio::time::sleep(backoff).await;
                    } else {
                        return Err(e);
                    }
                }
            }
        }
    }

    pub async fn create_response_steam(
        context: Arc<dyn AsyncEngineContext>,
        info: ConnectionInfo,
    ) -> Result<StreamSender> {
        let info =
            TcpStreamConnectionInfo::try_from(info).context("tcp-stream-connection-info-error")?;
        tracing::trace!("Creating response stream for {:?}", info);

        if info.stream_type != StreamType::Response {
            return Err(error!(
                "Invalid stream type; TcpClient requires the stream type to be `response`; however {:?} was passed",
                info.stream_type
            ));
        }

        if info.context != context.id() {
            return Err(error!(
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

        let reader_task = tokio::spawn(handle_reader(framed_reader, context, alive_tx));

        // transport specific handshake message
        let handshake = CallHomeHandshake {
            subject: info.subject,
            stream_type: StreamType::Response,
        };

        let handshake_bytes = match serde_json::to_vec(&handshake) {
            Ok(hb) => hb,
            Err(err) => {
                return Err(error!(
                    "create_response_steam: Error converting CallHomeHandshake to JSON array: {err:#}"
                ));
            }
        };
        let msg = TwoPartMessage::from_header(handshake_bytes.into());

        // issue the the first tcp handshake message
        framed_writer
            .send(msg)
            .await
            .map_err(|e| error!("failed to send handshake: {:?}", e))?;

        // set up the channel to send bytes to the transport layer
        let (bytes_tx, bytes_rx) = tokio::sync::mpsc::channel(16);

        // forwards the bytes send from this stream to the transport layer; hold the alive_rx half of the oneshot channel

        let writer_task = tokio::spawn(handle_writer(framed_writer, bytes_rx, alive_rx));

        tokio::spawn(async move {
            // await both tasks
            let (reader, writer) = tokio::join!(reader_task, writer_task);

            match (reader, writer) {
                (Ok(reader), Ok(writer)) => {
                    let reader = reader.into_inner();
                    let writer = writer.into_inner();

                    let mut stream = reader.unsplit(writer);

                    // close the stream
                    Ok(stream.shutdown().await?)
                }
                _ => {
                    tracing::error!("failed to join reader and writer tasks");
                    anyhow::bail!("failed to join reader and writer tasks");
                }
            }
        });

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

async fn handle_reader(
    framed_reader: FramedRead<tokio::io::ReadHalf<tokio::net::TcpStream>, TwoPartCodec>,
    context: Arc<dyn AsyncEngineContext>,
    alive_tx: tokio::sync::oneshot::Sender<()>,
) -> FramedRead<tokio::io::ReadHalf<tokio::net::TcpStream>, TwoPartCodec> {
    let mut framed_reader = framed_reader;
    let mut alive_tx = alive_tx;
    loop {
        tokio::select! {
            msg = framed_reader.next() => {
                match msg {
                    Some(Ok(two_part_msg)) => {
                        match two_part_msg.optional_parts() {
                           (Some(bytes), None) => {
                                let msg: ControlMessage = serde_json::from_slice(bytes).unwrap();
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
                                // not a control message, so we just continue
                               continue;
                           }
                        }
                    }
                    Some(Err(_)) => {
                        break;
                        // panic!("failed to decode message from stream: {:?}", e);
                        // break;
                    }
                    None => {
                        // let mut writer = framed_reader.into_inner();
                        // if let Err(e) = writer.shutdown().await {
                        //     tracing::trace!("failed to shutdown reader: {:?}", e);
                        // }
                        break;
                    }
                }
            }
            _ = alive_tx.closed() => {
                // the channel was closed, we should stop the stream
                break;
            }
        }
    }
    framed_reader
}

async fn handle_writer(
    mut framed_writer: FramedWrite<tokio::io::WriteHalf<tokio::net::TcpStream>, TwoPartCodec>,
    mut bytes_rx: tokio::sync::mpsc::Receiver<TwoPartMessage>,
    alive_rx: tokio::sync::oneshot::Receiver<()>,
) -> FramedWrite<tokio::io::WriteHalf<tokio::net::TcpStream>, TwoPartCodec> {
    while let Some(msg) = bytes_rx.recv().await {
        if let Err(e) = framed_writer.send(msg).await {
            tracing::trace!(
                "failed to send message to stream; possible disconnect: {:?}",
                e
            );

            // TODO - possibly propagate the error upstream
            break;
        }
    }
    drop(alive_rx);
    // if let Err(e) = framed_writer.get_mut().shutdown().await {
    //     eprintln!("failed to shutdown writer: {:?}", e);
    //     tracing::trace!("failed to shutdown writer: {:?}", e);
    // }
    framed_writer
}
