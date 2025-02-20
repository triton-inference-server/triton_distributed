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

use std::{fmt::Debug, future::Future, pin::Pin, sync::Arc};

pub use async_trait::async_trait;
use futures::stream::Stream;

/// All [`Send`] + [`Sync`] + `'static` types can be used as [`AsyncEngine`] request and response types.
pub trait Data: Send + Sync + 'static {}
impl<T: Send + Sync + 'static> Data for T {}

/// [`DataStream`] is a type alias for a stream of [`Data`] items. This can be adapted to a [`ResponseStream`]
/// by associating it with a [`AsyncEngineContext`].
pub type DataUnary<T> = Pin<Box<dyn Future<Output = T> + Send + Sync>>;
pub type DataStream<T> = Pin<Box<dyn Stream<Item = T> + Send + Sync>>;

pub type Engine<Req, Resp, E> = Arc<dyn AsyncEngine<Req, Resp, E>>;
pub type EngineUnary<Resp> = Pin<Box<dyn AsyncEngineUnary<Resp>>>;
pub type EngineStream<Resp> = Pin<Box<dyn AsyncEngineStream<Resp>>>;
pub type Context = Arc<dyn AsyncEngineContext>;

impl<T: Data> From<EngineStream<T>> for DataStream<T> {
    fn from(stream: EngineStream<T>) -> Self {
        Box::pin(stream)
    }
}

// The Controller and the Context when https://github.com/rust-lang/rust/issues/65991 becomes stable
pub trait AsyncEngineController: Send + Sync {}

/// The [`AsyncEngineContext`] trait defines the interface to control the resulting stream
/// produced by the engine.
#[async_trait]
pub trait AsyncEngineContext: Send + Sync + Debug {
    /// Unique ID for the Stream
    fn id(&self) -> &str;

    /// Returns true if `stop_generating()` has been called; otherwise, false.
    fn is_stopped(&self) -> bool;

    /// Returns true if `kill()` has been called; otherwise, false.
    /// This can be used with a `.take_while()` stream combinator to immediately terminate
    /// the stream.
    ///
    /// An ideal location for a `[.take_while(!ctx.is_killed())]` stream combinator is on
    /// the most downstream  return stream.
    fn is_killed(&self) -> bool;

    /// Calling this method when [`AsyncEngineContext::is_stopped`] is `true` will return
    /// immediately; otherwise, it will [`AsyncEngineContext::is_stopped`] will return true.
    async fn stopped(&self);

    /// Calling this method when [`AsyncEngineContext::is_killed`] is `true` will return
    /// immediately; otherwise, it will [`AsyncEngineContext::is_killed`] will return true.
    async fn killed(&self);

    // Controller

    /// Informs the [`AsyncEngine`] to stop producing results for this particular stream.
    /// This method is idempotent. This method does not invalidate results current in the
    /// stream. It might take some time for the engine to stop producing results. The caller
    /// can decided to drain the stream or drop the stream.
    fn stop_generating(&self);

    /// See [`AsyncEngineContext::stop_generating`].
    fn stop(&self);

    /// Extends the [`AsyncEngineContext::stop_generating`] also indicates a preference to
    /// terminate without draining the remaining items in the stream. This is implementation
    /// specific and may not be supported by all engines.
    fn kill(&self);
}

pub trait AsyncEngineContextProvider: Send + Sync + Debug {
    fn context(&self) -> Arc<dyn AsyncEngineContext>;
}

pub trait AsyncEngineUnary<Resp: Data>:
    Future<Output = Resp> + AsyncEngineContextProvider + Send + Sync
{
}

pub trait AsyncEngineStream<Resp: Data>:
    Stream<Item = Resp> + AsyncEngineContextProvider + Send + Sync
{
}

/// Engine is a trait that defines the interface for a steaming LLM completion engine.
/// The synchronous Engine version is does not need to be awaited.
#[async_trait]
pub trait AsyncEngine<Req: Data, Resp: Data + AsyncEngineContextProvider, E: Data>:
    Send + Sync
{
    /// Generate a stream of completion responses.
    async fn generate(&self, request: Req) -> Result<Resp, E>;
}

/// Adapter for a [`DataStream`] to a [`ResponseStream`].
///
/// A common pattern is to consume the [`ResponseStream`] with standard stream combinators
/// which produces a [`DataStream`] stream, then form a [`ResponseStream`] by propagating the
/// original [`AsyncEngineContext`].
pub struct ResponseStream<R: Data> {
    stream: DataStream<R>,
    ctx: Arc<dyn AsyncEngineContext>,
}

impl<R: Data> ResponseStream<R> {
    pub fn new(stream: DataStream<R>, ctx: Arc<dyn AsyncEngineContext>) -> Pin<Box<Self>> {
        Box::pin(Self { stream, ctx })
    }
}

impl<R: Data> Stream for ResponseStream<R> {
    type Item = R;

    #[inline]
    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        Pin::new(&mut self.stream).poll_next(cx)
    }
}

impl<R: Data> AsyncEngineStream<R> for ResponseStream<R> {}

impl<R: Data> AsyncEngineContextProvider for ResponseStream<R> {
    fn context(&self) -> Arc<dyn AsyncEngineContext> {
        self.ctx.clone()
    }
}

impl<R: Data> Debug for ResponseStream<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResponseStream")
            // todo: add debug for stream - possibly propagate some information about what
            // engine created the stream
            // .field("stream", &self.stream)
            .field("ctx", &self.ctx)
            .finish()
    }
}

impl<T: Data> AsyncEngineContextProvider for Pin<Box<dyn AsyncEngineUnary<T>>> {
    fn context(&self) -> Arc<dyn AsyncEngineContext> {
        AsyncEngineContextProvider::context(&**self)
    }
}

impl<T: Data> AsyncEngineContextProvider for Pin<Box<dyn AsyncEngineStream<T>>> {
    fn context(&self) -> Arc<dyn AsyncEngineContext> {
        AsyncEngineContextProvider::context(&**self)
    }
}
