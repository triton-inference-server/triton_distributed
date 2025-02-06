//! # Triton LLM Protocols
//!
//! This module contains the protocols, i.e. messages formats, used to exchange requests and responses
//! both publically via the HTTP API and internally between Triton components.
//!

use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use std::pin::Pin;

pub mod codec;
pub mod common;
pub mod openai;

/// The token ID type
pub type TokenIdType = u32;
pub type DataStream<T> = Pin<Box<dyn Stream<Item = T> + Send + Sync>>;

// TODO: This is an awkward dependency that we need to address
// Originally, all the Annotated/SSE Codec bits where in the LLM protocol module; however, [Annotated]
// has become the common response envelope for triton-distributed.
// We may want to move the original Annotated back here and has a Infallible conversion to the the
// ResponseEnvelop in triton-distributed.
pub use triton_distributed::protocols::annotated::Annotated;

/// The LLM responses have multiple different fields and nests of objects to get to the actual
/// text completion returned. This trait can be applied to the `choice` level objects to extract
/// the completion text.
///
/// To avoid an optional, if no completion text is found, the [`ContentProvider::content`] should
/// return an empty string.
pub trait ContentProvider {
    fn content(&self) -> String;
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Usage {
    pub prompt_tokens: i32,
    pub completion_tokens: i32,
    pub total_tokens: i32,
}

/// Converts of a stream of [codec::Message]s into a stream of [Annotated]s.
pub fn convert_sse_stream<R>(
    stream: DataStream<Result<codec::Message, codec::SseCodecError>>,
) -> DataStream<Annotated<R>>
where
    R: for<'de> Deserialize<'de> + Serialize,
{
    let stream = stream.map(|message| match message {
        Ok(message) => {
            let delta = Annotated::<R>::try_from(message);
            match delta {
                Ok(delta) => delta,
                Err(e) => Annotated::from_error(e.to_string()),
            }
        }
        Err(e) => Annotated::from_error(e.to_string()),
    });
    Box::pin(stream)
}
