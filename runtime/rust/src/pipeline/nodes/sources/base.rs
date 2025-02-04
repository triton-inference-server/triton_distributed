/*
 * Copyright 2024-2025 NVIDIA CORPORATION & AFFILIATES
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License. You may obtain a copy of
 * the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

use crate::engine::AsyncEngineContextProvider;

use super::*;

impl<In: PipelineIO, Out: PipelineIO> Default for Frontend<In, Out> {
    fn default() -> Self {
        Self {
            edge: OnceLock::new(),
            sinks: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

#[async_trait]
impl<In: PipelineIO, Out: PipelineIO> Source<In> for Frontend<In, Out> {
    async fn on_next(&self, data: In, _: private::Token) -> Result<(), Error> {
        self.edge
            .get()
            .ok_or(PipelineError::NoEdge)?
            .write(data)
            .await
    }

    fn set_edge(&self, edge: Edge<In>, _: private::Token) -> Result<(), PipelineError> {
        self.edge
            .set(edge)
            .map_err(|_| PipelineError::EdgeAlreadySet)?;
        Ok(())
    }
}

#[async_trait]
impl<In: PipelineIO, Out: PipelineIO + AsyncEngineContextProvider> Sink<Out> for Frontend<In, Out> {
    async fn on_data(&self, data: Out, _: private::Token) -> Result<(), Error> {
        let ctx = data.context();

        let mut sinks = self.sinks.lock().unwrap();
        let tx = sinks
            .remove(ctx.id())
            .ok_or(PipelineError::DetatchedStreamReceiver)
            .inspect_err(|_| {
                ctx.stop_generating();
            })?;
        drop(sinks);

        Ok(tx
            .send(data)
            .map_err(|_| PipelineError::DetatchedStreamReceiver)
            .inspect_err(|_| {
                ctx.stop_generating();
            })?)
    }
}

#[async_trait]
impl<In: PipelineIO, Out: PipelineIO> AsyncEngine<In, Out, Error> for Frontend<In, Out> {
    async fn generate(&self, request: In) -> Result<Out, Error> {
        let (tx, rx) = oneshot::channel::<Out>();
        {
            let mut sinks = self.sinks.lock().unwrap();
            sinks.insert(request.id().to_string(), tx);
        }
        self.on_next(request, private::Token {}).await?;
        Ok(rx.await.map_err(|_| PipelineError::DetatchedStreamSender)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::{error::PipelineErrorExt, ManyOut, SingleIn};

    #[tokio::test]
    async fn test_frontend_no_edge() {
        let source = Frontend::<SingleIn<()>, ManyOut<()>>::default();
        let error = source
            .generate(().into())
            .await
            .unwrap_err()
            .try_into_pipeline_error()
            .unwrap();

        match error {
            PipelineError::NoEdge => (),
            _ => panic!("Expected NoEdge error"),
        }

        let result = source
            .on_next(().into(), private::Token)
            .await
            .unwrap_err()
            .try_into_pipeline_error()
            .unwrap();

        match result {
            PipelineError::NoEdge => (),
            _ => panic!("Expected NoEdge error"),
        }
    }
}
