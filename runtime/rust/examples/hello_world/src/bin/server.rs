use hello_world::DEFAULT_NAMESPACE;
use std::sync::Arc;
use triton_distributed::{
    pipeline::{
        async_trait, network::Ingress, AsyncEngine, AsyncEngineContextProvider, Error, ManyOut,
        ResponseStream, SingleIn,
    },
    protocols::annotated::Annotated,
    stream, DistributedRuntime, Result, Runtime, Worker,
};

fn main() -> Result<()> {
    env_logger::init();
    let worker = Worker::from_settings()?;
    worker.execute(app)
}

async fn app(runtime: Runtime) -> Result<()> {
    let distributed = DistributedRuntime::from_settings(runtime.clone()).await?;
    backend(distributed).await
}

struct RequestHandler {}

impl RequestHandler {
    fn new() -> Arc<Self> {
        Arc::new(Self {})
    }
}

#[async_trait]
impl AsyncEngine<SingleIn<String>, ManyOut<Annotated<String>>, Error> for RequestHandler {
    async fn generate(&self, input: SingleIn<String>) -> Result<ManyOut<Annotated<String>>> {
        let (data, ctx) = input.into_parts();

        let chars = data
            .chars()
            .map(|c| Annotated::from_data(c.to_string()))
            .collect::<Vec<_>>();

        let stream = stream::iter(chars);

        Ok(ResponseStream::new(Box::pin(stream), ctx.context()))
    }
}

async fn backend(runtime: DistributedRuntime) -> Result<()> {
    // attach an ingress to an engine
    let ingress = Ingress::for_engine(RequestHandler::new())?;

    // // make the ingress discoverable via a component service
    // // we must first create a service, then we can attach one more more endpoints
    runtime
        .namespace(DEFAULT_NAMESPACE)?
        .component("backend")?
        .service_builder()
        .create()
        .await?
        .endpoint("generate")
        .endpoint_builder()
        .handler(ingress)
        .start()
        .await
}
