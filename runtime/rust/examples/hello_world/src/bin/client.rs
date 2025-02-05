use hello_world::DEFAULT_NAMESPACE;
use triton_distributed::{
    protocols::annotated::Annotated, stream::StreamExt, DistributedRuntime, Result, Runtime, Worker,
};

fn main() -> Result<()> {
    env_logger::init();
    let worker = Worker::from_settings()?;
    worker.execute(app)
}

async fn app(runtime: Runtime) -> Result<()> {
    let distributed = DistributedRuntime::from_settings(runtime.clone()).await?;

    let client = distributed
        .namespace(DEFAULT_NAMESPACE)?
        .component("backend")?
        .endpoint("generate")
        .client::<String, Annotated<String>>()
        .await?;

    client.wait_for_endpoints().await?;

    let mut stream = client.random("hello world".to_string().into()).await?;

    while let Some(resp) = stream.next().await {
        println!("{:?}", resp);
    }

    runtime.shutdown();

    Ok(())
}
