use async_trait::async_trait;

use super::*;

use crate::icp::events::EventPlane;

#[async_trait]
impl EventPlane for Namespace {
    fn subject(&self) -> String {
        format!("namespace.{}", self.name)
    }

    async fn publish(
        &self,
        event_name: impl AsRef<str> + Send + Sync,
        event: &(impl Serialize + Send + Sync),
    ) -> Result<()> {
        let bytes = serde_json::to_vec(event)?;
        self.publish_bytes(event_name, bytes).await
    }

    async fn publish_bytes(
        &self,
        event_name: impl AsRef<str> + Send + Sync,
        bytes: Vec<u8>,
    ) -> Result<()> {
        let subject = format!("{}.{}", self.subject(), event_name.as_ref());
        Ok(self
            .drt()
            .nats_client()
            .client()
            .publish(subject, bytes.into())
            .await?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // todo - make a distributed runtime fixture
    // todo - two options - fully mocked or integration test
    #[cfg(feature = "integration-tests")]
    #[tokio::test]
    async fn test_publish() {
        // todo - use rtest - make fixtures
        let dtr = DistributedRuntime::from_settings(Runtime::single_threaded().unwrap())
            .await
            .unwrap();

        let ns = dtr.namespace("test".to_string()).unwrap();

        ns.publish("test", &"test".to_string()).await.unwrap();
    }
}
