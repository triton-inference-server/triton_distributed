#![allow(dead_code)]
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
#![allow(unused_variables)]

use anyhow::Result;
use tracing as log;

use super::*;

use async_nats::client::Client;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum RequestType {
    SingleIn,
    ManyIn,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ResponseType {
    SingleOut,
    ManyOut,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RequestControlMessage {
    id: String,
    request_type: RequestType,
    response_type: ResponseType,
    connection_info: ConnectionInfo,
}

pub type PushRouter<In, Out> =
    Arc<dyn AsyncEngine<SingleIn<AddressedRequest<In>>, ManyOut<Out>, Error>>;

pub struct AddressedRequest<T> {
    request: T,
    address: String,
}

impl<T> AddressedRequest<T> {
    pub fn new(request: T, address: String) -> Self {
        Self { request, address }
    }

    fn into_parts(self) -> (T, String) {
        (self.request, self.address)
    }
}

pub struct AddressedPushRouter {
    // todo: generalize with a generic
    req_transport: Client,

    // todo: generalize with a generic
    resp_transport: Arc<tcp::server::TcpStreamServer>,
}

impl AddressedPushRouter {
    pub fn new(
        req_transport: Client,
        resp_transport: Arc<tcp::server::TcpStreamServer>,
    ) -> Result<Arc<Self>> {
        Ok(Arc::new(Self {
            req_transport,
            resp_transport,
        }))
    }
}

#[async_trait]
impl<T, U> AsyncEngine<SingleIn<AddressedRequest<T>>, ManyOut<U>, Error> for AddressedPushRouter
where
    T: Data + Serialize,
    U: Data + for<'de> Deserialize<'de>,
{
    async fn generate(&self, request: SingleIn<AddressedRequest<T>>) -> Result<ManyOut<U>, Error> {
        let id = request.context().id().to_string();
        let (addressed_request, context) = request.transfer(());
        let (request, address) = addressed_request.into_parts();
        let engine_ctx = context.context();

        // registration options for the data plane in a singe in / many out configuration
        let options = StreamOptions::builder()
            .context(engine_ctx.clone())
            .enable_request_stream(false)
            .enable_response_stream(true)
            .build()
            .unwrap();

        // register our needs with the data plane
        // todo - generalize this with a generic data plane object which hides the specific transports
        let pending_connections: PendingConnections = self.resp_transport.register(options).await;

        // validate and unwrap the RegisteredStream object
        let pending_response_stream = match pending_connections.into_parts() {
            (None, Some(recv_stream)) => recv_stream,
            _ => {
                panic!("Invalid data plane registration for a SingleIn/ManyOut transport");
            }
        };

        // separate out the the connection info and the stream provider from the registered stream
        let (connection_info, response_stream_provider) = pending_response_stream.into_parts();

        // package up the connection info as part of the "header" component of the two part message
        // used to issue the request on the
        // todo -- this object should be automatically created by the register call, and achieved by to the two into_parts()
        // calls. all the information here is provided by the [`StreamOptions`] object and/or the dataplane object
        let control_message = RequestControlMessage {
            id: engine_ctx.id().to_string(),
            request_type: RequestType::SingleIn,
            response_type: ResponseType::ManyOut,
            connection_info,
        };

        // next build the two part message where we package the connection info and the request into
        // a single Vec<u8> that can be sent over the wire.
        // --- package this up in the WorkQueuePublisher ---
        let ctrl = serde_json::to_vec(&control_message).unwrap();
        let data = serde_json::to_vec(&request).unwrap();

        log::trace!(
            "[req: {}] packaging two-part message; ctrl: {} bytes, data: {} bytes",
            id,
            ctrl.len(),
            data.len()
        );

        let msg = TwoPartMessage::from_parts(ctrl.into(), data.into());

        // the request plane / work queue shoudl provide a two part message codec that can be used
        // or it should take a two part message directly
        // todo - update this
        let codec = TwoPartCodec::default();
        let buffer = codec.encode_message(msg).unwrap();

        // TRANSPORT ABSTRACT REQUIRED - END HERE

        log::trace!("[req: {}] enqueueing two-part message to nats", id);

        // we might need to add a timeout on this if there is no subscriber to the subject; however, I think nats
        // will handle this for us
        let response = self
            .req_transport
            .request(address.to_string(), buffer)
            .await
            .unwrap();

        log::trace!("[req: {}] awaiting transport handshake", id);
        let response_stream = response_stream_provider
            .await
            .map_err(|_| PipelineError::DetatchedStreamReceiver)?
            .map_err(PipelineError::ConnectionFailed)?;

        let stream = tokio_stream::wrappers::ReceiverStream::new(response_stream.rx);

        let stream = stream.map(|msg| {
            let resp: U = serde_json::from_slice(&msg).unwrap();
            resp
        });

        Ok(ResponseStream::new(Box::pin(stream), engine_ctx))
    }
}
