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
use tracing as log;

/// Add bingings from this crate to the provided module
pub fn add_to_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PythonAsyncEngine>()?;
    Ok(())
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
    generator: PyObject,
    event_loop: PyObject,
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
            generator,
            event_loop,
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

        let id = context.id().to_string();
        log::trace!("processing request: {}", id);

        // Clone the PyObject to move into the thread

        // Create a channel to communicate between the Python thread and the Rust async context
        let (tx, rx) = mpsc::channel::<Annotated<Resp>>(128);
        let tx_error = tx.clone();

        let stream = Python::with_gil(|py| {
            let py_request = pythonize(py, &request)?;
            let gen = self.generator.call1(py, (py_request,))?;
            let locals = TaskLocals::new(self.event_loop.bind(py).clone());
            pyo3_async_runtimes::tokio::into_stream_with_locals_v1(locals, gen.into_bound(py))
        })?;

        let stream = Box::pin(stream);

        let process = |item: Result<Py<PyAny>, PyErr>| -> Result<Annotated<Resp>, Error> {
            let item = item
                .map_err(|err| error!("error processing python async generator stream: {}", err))?;

            let response = Python::with_gil(|py| depythonize::<Resp>(&item.into_bound(py)))?;
            let response = Annotated::from_data(response);

            Ok(response)
        };

        // process the stream
        // any error thrown in the stream will be caught and complete the processing task
        // errors are captured by a task that is watching the processing task
        // the error will be emitted as an annotated error
        let processor = tokio::spawn(async move {
            log::trace!("processing stream from python async generator: {}", id);
            let mut stream = stream;

            while let Some(item) = stream.next().await {
                // let mut done = false;
                let response = match process(item) {
                    Ok(response) => response,
                    Err(err) => {
                        // done = true;
                        Annotated::from_error(err.to_string())
                    }
                };

                if tx.send(response).await.is_err() {
                    log::error!("generator response channel was dropped: {}", id);
                    return Err(error!("generator response channel was dropped"));
                }

                // if done {
                //     break;
                // }
            }

            Result::<()>::Ok(())
        });

        tokio::spawn(async move {
            match processor.await {
                Ok(Ok(_)) => {}
                Ok(Err(err)) => {
                    log::error!("error processing python async generator: {}", err);
                    tx_error
                        .send(Annotated::from_error(err.to_string()))
                        .await
                        .unwrap();
                }
                Err(err) => {
                    log::error!(
                        "error on tokio task for processing python async generator stream: {}",
                        err
                    );
                    tx_error
                        .send(Annotated::from_error(err.to_string()))
                        .await
                        .unwrap();
                }
            }
        });

        let stream = ReceiverStream::new(rx);

        Ok(ResponseStream::new(Box::pin(stream), context.context()))
    }
}
