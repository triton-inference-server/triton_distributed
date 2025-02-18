use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

use crate::Result;

// #[async_trait]
// pub trait Publisher: Debug + Clone + Send + Sync {
//     async fn publish(&self, event: &(impl Serialize + Send + Sync)) -> Result<()>;
// }

/// A [EventPlane] is a component that can publish and/or subscribe to events on the event plane.
///
/// Each implementation of [EventPlane] will define the root subject.
#[async_trait]
pub trait EventPlane {
    /// The base subject used for this implementation of the [EventPlane].
    fn subject(&self) -> String;

    /// Publish a single event to the event plane. The `event_name` will be `.` concatenated with the
    /// base subject provided by the implementation.
    async fn publish(
        &self,
        event_name: impl AsRef<str> + Send + Sync,
        event: &(impl Serialize + Send + Sync),
    ) -> Result<()>;

    /// Publish a single event as bytes to the event plane. The `event_name` will be `.` concatenated with the
    /// base subject provided by the implementation.
    async fn publish_bytes(
        &self,
        event_name: impl AsRef<str> + Send + Sync,
        bytes: Vec<u8>,
    ) -> Result<()>;

    // /// Create a new publisher for the given event name. The `event_name` will be `.` concatenated with the
    // /// base subject provided by the implementation.
    // fn publisher(&self, event_name: impl AsRef<str>) -> impl Publisher;

    // /// Create a new publisher for the given event name. The `event_name` will be `.` concatenated with the
    // fn publisher(&self, event_name: impl AsRef<str>) -> Result<Publisher>;
    // fn publisher_bytes(&self, event_name: impl AsRef<str>) -> &PublisherBytes;
}
