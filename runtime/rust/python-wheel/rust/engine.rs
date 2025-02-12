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

pub use serde::{Deserialize, Serialize};
pub use triton_distributed::{
    error,
    pipeline::{
        async_trait, AsyncEngine, AsyncEngineContextProvider, Data, ManyOut, ResponseStream,
        SingleIn,
    },
    protocols::annotated::Annotated,
    Error, Result,
};

use pyo3::prelude::*;
use pyo3_async_runtimes::TaskLocals;
use pythonize::{depythonize, pythonize};

use tokio::sync::mpsc;
use tokio_stream::{wrappers::ReceiverStream, StreamExt};

/// Add bingings from this crate to the provided module
pub fn add_to_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PythonAsyncEngine>()?;
    Ok(())
}

#[derive(Debug, thiserror::Error)]
enum ResponseProcessingError {
    #[error("python exception: {0}")]
    PythonException(String),

    #[error("deserialize error: {0}")]
    DeserializeError(String),

    #[error("gil offload error: {0}")]
    OffloadError(String),
}

// todos:
// - [ ] enable context cancellation
//   - this will likely require a change to the function signature python calling arguments
// - [ ] rename `PythonAsyncEngine` to `PythonServerStreamingEngine` to be more descriptive
// - [ ] other `AsyncEngine` implementations will have a similar pattern, i.e. one AsyncEngine
//       implementation per struct

/// Rust/Python bridge that maps to the [`AsyncEngine`] trait
///
/// Currently this is only implemented for the [`SingleIn`] and [`ManyOut`] types; however,
/// more [`AsyncEngine`] implementations can be added in the future.
///
/// For the [`SingleIn`] and [`ManyOut`] case, this implementation will take a Python async
/// generator and convert it to a Rust async stream.
///
/// ```python
/// class ComputeEngine:
///     def __init__(self):
///         self.compute_engine = make_compute_engine()
///
///     def generate(self, request):
///         async generator():
///            async for output in self.compute_engine.generate(request):
///                yield output
///         return generator()
///
/// def main():
///     loop = asyncio.create_event_loop()
///     compute_engine = ComputeEngine()
///     engine = PythonAsyncEngine(compute_engine.generate, loop)
///     service = RustService()
///     service.add_engine("model_name", engine)
///     loop.run_until_complete(service.run())
/// ```
#[pyclass]
#[derive(Clone)]
pub struct PythonAsyncEngine {
    generator: Arc<PyObject>,
    event_loop: Arc<PyObject>,
}

#[pymethods]
impl PythonAsyncEngine {
    /// Create a new instance of the PythonAsyncEngine
    ///
    /// # Arguments
    /// - `generator`: a Python async generator that will be used to generate responses
    /// - `event_loop`: the Python event loop that will be used to run the generator
    ///
    /// Note: In Rust land, the request and the response are both concrete; however, in
    /// Python land, the request and response not strongly typed, meaning the generator
    /// could accept a different type of request or return a different type of response
    /// and we would not know until runtime.
    #[new]
    pub fn new(generator: PyObject, event_loop: PyObject) -> PyResult<Self> {
        Ok(PythonAsyncEngine {
            generator: Arc::new(generator),
            event_loop: Arc::new(event_loop),
        })
    }
}

#[async_trait]
impl<Req, Resp> AsyncEngine<SingleIn<Req>, ManyOut<Annotated<Resp>>, Error> for PythonAsyncEngine
where
    Req: Data + Serialize,
    Resp: Data + for<'de> Deserialize<'de>,
{
    async fn generate(&self, request: SingleIn<Req>) -> Result<ManyOut<Annotated<Resp>>, Error> {
        // Create a context
        let (request, context) = request.transfer(());
        let ctx = context.context();

        let id = context.id().to_string();
        tracing::trace!("processing request: {}", id);

        // Clone the PyObject to move into the thread

        // Create a channel to communicate between the Python thread and the Rust async context
        let (tx, rx) = mpsc::channel::<Annotated<Resp>>(128);

        let generator = self.generator.clone();
        let event_loop = self.event_loop.clone();

        // Acquiring the GIL is similar to acquiring a standard lock/mutex
        // Performing this in an tokio async task could block the thread for an undefined amount of time
        // To avoid this, we spawn a blocking task to acquire the GIL and perform the operations needed
        // while holding the GIL.
        //
        // Under low GIL contention, we wouldn't need to do this.
        // However, under high GIL contention, this can lead to significant performance degradation.
        //
        // Since we cannot predict the GIL contention, we will always use the blocking task and pay the
        // cost. The Python GIL is the gift that keeps on giving -- performance hits...
        let stream = tokio::task::spawn_blocking(move || {
            Python::with_gil(|py| {
                let py_request = pythonize(py, &request)?;
                let gen = generator.call1(py, (py_request,))?;
                let locals = TaskLocals::new(event_loop.bind(py).clone());
                pyo3_async_runtimes::tokio::into_stream_with_locals_v1(locals, gen.into_bound(py))
            })
        })
        .await??;

        let stream = Box::pin(stream);

        // process the stream
        // any error thrown in the stream will be caught and complete the processing task
        // errors are captured by a task that is watching the processing task
        // the error will be emitted as an annotated error
        let request_id = id.clone();

        tokio::spawn(async move {
            tracing::debug!(
                request_id,
                "starting task to process python async generator stream"
            );

            let mut stream = stream;
            let mut count = 0;

            while let Some(item) = stream.next().await {
                count += 1;
                tracing::trace!(
                    request_id,
                    "processing the {}th item from python async generator",
                    count
                );

                let mut done = false;

                let response = match process_item::<Resp>(item).await {
                    Ok(response) => response,
                    Err(e) => {
                        done = true;

                        let msg = match &e {
                            ResponseProcessingError::DeserializeError(e) => {
                                // tell the python async generator to stop generating
                                // right now, this is impossible as we are not passing the context to the python async generator
                                // todo: add task-local context to the python async generator
                                // see: https://github.com/triton-inference-server/triton_distributed/issues/130
                                ctx.stop_generating();
                                let msg = format!("critical error: invalid response object from python async generator; application-logic-mismatch: {}", e);
                                tracing::error!(request_id, "{}", msg);
                                msg
                            }
                            ResponseProcessingError::PythonException(e) => {
                                let msg = format!("a python exception was caught while processing the async generator: {}", e);
                                tracing::warn!(request_id, "{}", msg);
                                msg
                            }
                            ResponseProcessingError::OffloadError(e) => {
                                let msg = format!("critical error: failed to offload the python async generator to a new thread: {}", e);
                                tracing::error!(request_id, "{}", msg);
                                msg
                            }
                        };

                        Annotated::from_error(msg)
                    }
                };

                if tx.send(response).await.is_err() {
                    tracing::trace!(
                        request_id,
                        "error forwarding annotated response to channel; channel is closed"
                    );
                    break;
                }

                if done {
                    tracing::debug!(
                        request_id,
                        "early termination of python async generator stream task"
                    );
                    break;
                }
            }

            tracing::debug!(
                request_id,
                "finished processing python async generator stream"
            );
        });

        let stream = ReceiverStream::new(rx);

        Ok(ResponseStream::new(Box::pin(stream), context.context()))
    }
}

async fn process_item<Resp>(
    item: Result<Py<PyAny>, PyErr>,
) -> Result<Annotated<Resp>, ResponseProcessingError>
where
    Resp: Data + for<'de> Deserialize<'de>,
{
    let item = item.map_err(|e| ResponseProcessingError::PythonException(e.to_string()))?;

    let response = tokio::task::spawn_blocking(move || {
        Python::with_gil(|py| depythonize::<Resp>(&item.into_bound(py)))
    })
    .await
    .map_err(|e| ResponseProcessingError::OffloadError(e.to_string()))?
    .map_err(|e| ResponseProcessingError::DeserializeError(e.to_string()))?;

    let response = Annotated::from_data(response);

    Ok(response)
}
