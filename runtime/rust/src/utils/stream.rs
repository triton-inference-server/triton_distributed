use futures::stream::{Stream, StreamExt};
use std::{
    future::Future,
    pin::Pin,
    task::{Context, Poll},
};

use tokio::time::{self, sleep_until, Duration, Instant, Sleep};

pub struct DeadlineStream<S> {
    stream: S,
    sleep: Pin<Box<Sleep>>,
}

impl<S: Stream + Unpin> Stream for DeadlineStream<S> {
    type Item = S::Item;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // First, check if our sleep future has completed
        if let Poll::Ready(_) = Pin::new(&mut self.sleep).poll(cx) {
            // The deadline expired; end the stream now
            return Poll::Ready(None);
        }

        // Otherwise, poll the underlying stream
        self.as_mut().stream.poll_next_unpin(cx)
    }
}

pub fn until_deadline<S: Stream + Unpin>(stream: S, deadline: Instant) -> DeadlineStream<S> {
    DeadlineStream {
        stream,
        sleep: Box::pin(sleep_until(deadline)),
    }
}
