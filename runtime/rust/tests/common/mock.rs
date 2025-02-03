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

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use async_trait::async_trait;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use triton_distributed::engine::{AsyncEngine, AsyncEngineContext, Data, ResponseStream};
use triton_distributed::pipeline::{
    context::{Context, StreamContext},
    Error, ManyOut, PipelineError, PipelineIO, SegmentSource, SingleIn,
};

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum LatencyModel {
    NoDelay,
    ConstantDelayInNanos(u64),
    NormalDistributionInNanos(u64, u64),
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct MockNetworkOptions {
    request_latency: LatencyModel,
    response_latency: LatencyModel,
}

impl Default for MockNetworkOptions {
    fn default() -> Self {
        Self {
            request_latency: LatencyModel::NoDelay,
            response_latency: LatencyModel::NoDelay,
        }
    }
}

#[derive(Debug, Clone)]
struct ControlPlaneRequest {
    id: String,
    request: Vec<u8>,

    // convert this into an interface where it describes the worker address
    // and how to communicate with the worker
    resp_tx: mpsc::Sender<DataPlaneMessage>,
}

enum MockNetworkControlEvents {
    ControlPlaneRequest(ControlPlaneRequest),
    Cancel(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum MockNetworkDataPlaneHeaders {
    Handshake(Handshake),
    Error(String),

    // tells the subscriber that the stream has ended
    // not all transports will be sender side closable, therefore,
    // we need a way to signal the end of the stream
    //
    // note: for transports like nats where the subscriber could
    // be left dangling, we will also want to have a keep alive
    // and a timeout mechanism
    Sentinal,

    // heart beat / keep-alive signal to maintain the connection
    HeartBeat,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Status {
    Ok,
    Error(String),
}

// for transports that support headers, we will use headers for events and the body for the bytes
// for transports like tcp, we may send them as two separate messages on the same socket or as a single
// compound message like the [`DataEnvelope`] object below
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Handshake {
    request_id: String,
    worker_id: Option<String>,
    status: Status,
}

struct DataPlaneMessage {
    pub headers: Option<MockNetworkDataPlaneHeaders>,
    pub body: Vec<u8>,
}

/// This is an example transport that will inject latency into the response stream.
/// This mimics a network transport that has a delay in the response.
pub struct MockNetworkTransport<T: PipelineIO, U: PipelineIO> {
    req: std::marker::PhantomData<T>,
    resp: std::marker::PhantomData<U>,
}

impl<Req: PipelineIO, Resp: PipelineIO> MockNetworkTransport<Req, Resp> {
    pub fn new_egress_ingress(
        options: MockNetworkOptions,
    ) -> (
        Arc<MockNetworkEgress<Req, Resp>>,
        MockNetworkIngress<Req, Resp>,
    ) {
        let (ctrl_tx, ctrl_rx) = mpsc::channel::<MockNetworkControlEvents>(8);

        // construct the egress/request-sender/response-receiver
        let egress = Arc::new(MockNetworkEgress::<Req, Resp>::new(
            options.clone(),
            ctrl_tx.clone(),
        ));

        // construct the ingress/request-receiver/response-sender
        let ingress = MockNetworkIngress::<Req, Resp>::new(options.clone(), ctrl_rx);

        (egress, ingress)
    }
}

#[allow(dead_code)]
pub struct MockNetworkEgress<Req: PipelineIO, Resp: PipelineIO> {
    options: MockNetworkOptions,
    ctrl_tx: mpsc::Sender<MockNetworkControlEvents>,
    req: std::marker::PhantomData<Req>,
    resp: std::marker::PhantomData<Resp>,
}

impl<Req: PipelineIO, Resp: PipelineIO> MockNetworkEgress<Req, Resp> {
    fn new(options: MockNetworkOptions, ctrl_tx: mpsc::Sender<MockNetworkControlEvents>) -> Self {
        Self {
            options,
            ctrl_tx,
            req: std::marker::PhantomData,
            resp: std::marker::PhantomData,
        }
    }
}

#[async_trait]
impl<T: Data, U: Data> AsyncEngine<SingleIn<T>, ManyOut<U>, Error>
    for MockNetworkEgress<SingleIn<T>, ManyOut<U>>
where
    T: Data + Serialize,
    U: for<'de> Deserialize<'de> + Data,
{
    async fn generate(&self, request: SingleIn<T>) -> Result<ManyOut<U>, Error> {
        let id = request.id().to_string();

        // serialze the request
        let request = request.try_map(|req| serde_json::to_vec(&req))?;

        // transfer the request context to a stream context
        let (data, context) = request.transfer(());
        let context = Arc::new(StreamContext::from(context));

        // subscribe to the response stream
        // but in this case, we are doing a mock, so we are going to be more explicit
        // since we are transfering data over a channel instead of the networ, creating the channel
        // is the same as subscribing to the response stream
        let (data_tx, data_rx) = mpsc::channel::<DataPlaneMessage>(16);
        let mut byte_stream = tokio_stream::wrappers::ReceiverStream::new(data_rx);

        // prepare the stateful objects that will be used to monitor the response stream
        // finish_rx is a oneshot channel that will be used to signal the natural termination of the stream
        let (finished_tx, finished_rx) = tokio::sync::oneshot::channel::<()>();
        let stream_monitor = ResponseMonitor {
            ctx: context.clone(),
            finish_rx: finished_rx,
        };

        // create the control plane request
        // when this is issued, control is handed off to the control plane and the downstream segment
        // sometimes we might include the local server address and port for the response find its way home
        // todo(design) this will be part of the generalization error for multiple transport types
        let request = ControlPlaneRequest {
            id,
            request: data,
            resp_tx: data_tx,
        };

        // send the request to the control plane
        self.ctrl_tx
            .send(MockNetworkControlEvents::ControlPlaneRequest(request))
            .await
            .map_err(|e| PipelineError::ControlPlaneRequestError(e.to_string()))?;

        // the first message from the remote publisher on the data plane needs to be a handshake message
        // the handshake will indicate to what stream the data belongs to and if the remote segment was
        // able to process the request.
        //
        // note: in the case of the mock transport, the handshaking of the request id is not strictly
        // because the channel is specific to the request. this is similar to other transports like nats
        // where we will subscribe to a response stream on a subject unique to the stream.
        match byte_stream.next().await {
            Some(DataPlaneMessage { headers, body }) => {
                if !body.is_empty() {
                    Err(PipelineError::ControlPlaneRequestError(
                        "Expected an empty body for the handshake message".to_string(),
                    ))?;
                }
                match headers {
                    Some(header) => {
                        match header {
                            MockNetworkDataPlaneHeaders::Handshake(handshake) => {
                                match handshake.status {
                                    Status::Ok => {}
                                    Status::Error(e) => {
                                        // todo(metrics): increment metric counter for failed handshakes
                                        Err(PipelineError::ControlPlaneRequestError(format!(
                                            "remote segment was unable to process request: {}",
                                            e
                                        )))?;
                                    }
                                }
                            }
                            _ => {
                                Err(PipelineError::ControlPlaneRequestError(format!(
                                    "Expected a handshake message; got: {:?}",
                                    header
                                )))?;
                            }
                        }
                    }
                    _ => {
                        Err(PipelineError::ControlPlaneRequestError(
                            "Failed to receive properly formatted handshake on data plane"
                                .to_string(),
                        ))?;
                    }
                }
            }
            None => {
                // todo(metrics): increment metric counter for failed requests
                Err(PipelineError::ControlPlaneRequestError(
                    "Failed data plane connection closed before receieving handshake".to_string(),
                ))?;
            }
        }

        let decoded = byte_stream
            // .inspect(|_item| {
            //     // todo(metrics) increment the metrics counter by the number of bytes
            // })
            .scan(Some(stream_monitor), move |_stream_monitor, item| {
                // we could check the kill state of the context and terminate the stream here
                // if our transport needs a heartbeat, trigger a heartbeat here the monitor
                if let Some(headers) = &item.headers {
                    match headers {
                        MockNetworkDataPlaneHeaders::HeartBeat => {
                            // todo(metrics): increment metric counter for heartbeats
                            // send a heartbeat to the control plane
                            // this is a good place to send a heartbeat to the control plane
                            // to keep the connection alive
                        }
                        MockNetworkDataPlaneHeaders::Sentinal => {
                            // todo(metrics): increment metric counter for sentinals
                            // the stream has ended
                            // send a sentinal to the control plane
                            // this is a good place to send a sentinal to the control plane
                            // to indicate the end of the stream
                            return futures::future::ready(None);
                        }
                        _ => {}
                    }
                }

                futures::future::ready(Some(item))
            })
            // decode the response
            .map(move |item| {
                serde_json::from_slice::<U>(&item.body).expect("failed to deserialize response")
            });

        // cancellation can be tricky and is transport / protocol specific
        // in this case, our control plane channel for the is both orded and 1:1, thus we can
        // use that fact to first send the request, then forward any cancellation requests
        // this ensure the downstream node should register the context/request id before any
        // cancellation requests are sent

        // create the cancellation monitor object
        let cancellation_monitor = CancellationMonitor {
            ctx: context.clone(),
            ctrl_tx: self.ctrl_tx.clone(),
            finish_tx: finished_tx,
        };

        // launch the cancellation monitor task
        tokio::spawn(cancellation_monitor.execute());

        Ok(ResponseStream::new(Box::pin(decoded), context))
    }
}

/// For our MocNetworkTransport, the Ingress will be the one that will be receiving the requests
/// and pushes back the responses
///
/// As such, the Ingress will be the one that will be responsible for receiving control plane messages.
#[allow(dead_code)]
pub struct MockNetworkIngress<Req: PipelineIO, Resp: PipelineIO> {
    options: MockNetworkOptions,
    ctrl_rx: mpsc::Receiver<MockNetworkControlEvents>,
    segment: OnceLock<Arc<SegmentSource<Req, Resp>>>,
}

impl<Req: PipelineIO, Resp: PipelineIO> MockNetworkIngress<Req, Resp> {
    fn new(options: MockNetworkOptions, ctrl_rx: mpsc::Receiver<MockNetworkControlEvents>) -> Self {
        Self {
            options,
            ctrl_rx,
            segment: OnceLock::new(),
        }
    }

    pub fn segment(&self, segment: Arc<SegmentSource<Req, Resp>>) -> Result<(), PipelineError> {
        self.segment
            .set(segment)
            .map_err(|_| PipelineError::EdgeAlreadySet)
    }
}

impl<T: Data, U: Data> MockNetworkIngress<SingleIn<T>, ManyOut<U>>
where
    T: Data + for<'de> Deserialize<'de>,
    U: Data + Serialize,
{
    pub async fn execute(self) -> Result<(), PipelineError> {
        let mut state = HashMap::<String, Arc<dyn AsyncEngineContext>>::new();
        let worker_id = uuid::Uuid::new_v4().to_string();
        let mut ctrl_rx = self.ctrl_rx;
        let segment = self.segment.get().expect("segment not set").clone();

        while let Some(event) = ctrl_rx.recv().await {
            match event {
                MockNetworkControlEvents::ControlPlaneRequest(req) => {
                    // todo(metrics): increment metric counter for bytes received
                    // todo(metrics): increment metric counter for requests received
                    let id = req.id.clone();
                    tracing::debug!("[ingress] received request [id: {}]", id);

                    // deserialize the request
                    let request = serde_json::from_slice::<T>(&req.request)
                        .expect("failed to deserialize request");

                    // extend request with context
                    let request = Context::<T>::with_id(request, req.id.clone());

                    // create the response stream
                    let response = segment.generate(request).await;

                    let handshake = match &response {
                        Ok(_) => Handshake {
                            request_id: req.id,
                            worker_id: Some(worker_id.clone()),
                            status: Status::Ok,
                        },
                        Err(e) => Handshake {
                            request_id: req.id,
                            worker_id: Some(worker_id.clone()),
                            status: Status::Error(e.to_string()),
                        },
                    };

                    tracing::debug!("[ingress] sending handshake [id: {}]: {:?}", id, handshake);

                    // serialize the handshake
                    let handshake = DataPlaneMessage {
                        headers: Some(MockNetworkDataPlaneHeaders::Handshake(handshake)),
                        body: vec![],
                    };

                    // send the handshake
                    req.resp_tx
                        .send(handshake)
                        .await
                        .expect("failed to send handshake");

                    tracing::trace!("[ingress] handshake sent [id: {}]", id);

                    if let Ok(response) = response {
                        // spawn a task to process the response stream:
                        // - serialize each response
                        // - forward the bytes to the data plane
                        tracing::debug!("[ingress] processing response stream [id: {}]", id);

                        tokio::spawn(async move {
                            let mut response = response;
                            while let Some(resp) = response.next().await {
                                tracing::trace!("[ingress] received response [id: {}]", id);

                                let resp_bytes = serde_json::to_vec(&resp)
                                    .expect("failed to serialize response");

                                let msg = DataPlaneMessage {
                                    headers: None,
                                    body: resp_bytes,
                                };

                                // send the response
                                req.resp_tx
                                    .send(msg)
                                    .await
                                    .expect("failed to send response");

                                tracing::trace!("[ingress] sent response [id: {}]", id);
                            }

                            tracing::debug!("response stream completed [id: {}]", id);
                        });
                    }
                }
                MockNetworkControlEvents::Cancel(id) => {
                    // todo(metrics): increment metric counter for cancelled requests
                    // todo(metrics): increment metric counter for bytes received
                    // todo(metrics): increment metric counter for requests received

                    // cancel the request
                    if let Some(tx) = state.remove(&id) {
                        tx.stop_generating();
                    }
                }
            }
        }

        Ok(())
    }
}

// fn create_error_message(id: &str, e: &str) -> Hand {
//     format!("Failed to deserialize request [id: {}]: {}", id, e)
// }

/// Object transfered to the Cancellation Monitor Task
///
/// The cancellation monitor task will be responsible for taking action on a
/// cancellation request.
///
/// This object holds a oneshot channel that will be used to signal the natural
/// termination of the stream.
///
/// Our cancellation monitor task select on those two signals and complete when
/// either of them is completed.
struct CancellationMonitor {
    ctx: Arc<StreamContext>,

    // control plane sender
    ctrl_tx: tokio::sync::mpsc::Sender<MockNetworkControlEvents>,

    // the cancellation mni
    // as completed
    finish_tx: tokio::sync::oneshot::Sender<()>,
}

impl CancellationMonitor {
    async fn execute(self) {
        // select on the finish_rx and the kill signal
        let ctx = self.ctx;
        let ctrl_tx = self.ctrl_tx;
        let mut finish_tx = self.finish_tx;

        tokio::select! {
            _ = ctx.stopped() => {
                // todo(metrics): increment metric counter for cancelled requests
                // send a cancellation request to the control plane
                let _ = ctrl_tx.send(MockNetworkControlEvents::Cancel(ctx.id().to_string())).await;
            }
            _ = finish_tx.closed() => {
                // the stream has completed naturally
            }
        }
    }
}

// held by the scan combinator
#[allow(dead_code)]
struct ResponseMonitor {
    ctx: Arc<StreamContext>,
    finish_rx: tokio::sync::oneshot::Receiver<()>,
}
