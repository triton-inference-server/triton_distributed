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
        //tracing::debug!("Polling next item. Now: {:?}, Deadline: {:?}", Instant::now(), self.deadline);
        // Check if we've passed the deadline
        //if Instant::now() >= self.deadline {
        if Pin::new(&mut self.sleep).poll(cx).is_ready() {
            //tracing::debug!("Deadline expired. Now: {:?}, Deadline: {:?}", Instant::now(), self.deadline);
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
        // Set an async task that sleeps until deadline and wakes up to cancel the stream
        sleep: Box::pin(sleep_until(deadline)),
    }
}

#[cfg(test)]
mod tests {
    use futures::stream::{self, Stream, StreamExt};
    use tokio::pin;

    use super::*;

    #[tokio::test]
    async fn test_until_deadline_exceeded() {
        let stream = stream::iter(vec![100, 100, 200]);
        let stream = stream.then(|x| {
            let sleep = time::sleep(Duration::from_millis(x));
            async move {
                sleep.await;
                x
            }
        });
        let deadline = Instant::now() + Duration::from_millis(300);
        let mut result = Vec::new();
        pin!(stream);
        let mut stream = until_deadline(stream, deadline);
        while let Some(x) = stream.next().await {
            result.push(x);
        }
        assert_eq!(result, vec![100, 100]);
    }

    async fn test_until_complete_before_deadline() {
        let stream = stream::iter(vec![100, 100]);
        let stream = stream.then(|x| {
            let sleep = time::sleep(Duration::from_millis(x));
            async move {
                sleep.await;
                x
            }
        });
        let deadline = Instant::now() + Duration::from_millis(300);
        let mut result = Vec::new();
        pin!(stream);
        let mut stream = until_deadline(stream, deadline);
        while let Some(x) = stream.next().await {
            result.push(x);
        }
        assert_eq!(result, vec![100, 100]);
    }


}
