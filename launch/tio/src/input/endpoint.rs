// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

use triton_distributed_llm::{
    backend::Backend,
    http::service::discovery::ModelEntry,
    model_type::ModelType,
    preprocessor::OpenAIPreprocessor,
    types::{
        openai::chat_completions::{ChatCompletionResponseDelta, NvCreateChatCompletionRequest},
        Annotated,
    },
};
use triton_distributed_runtime::pipeline::{
    network::Ingress, ManyOut, Operator, SegmentSource, ServiceBackend, SingleIn, Source,
};
use triton_distributed_runtime::{protocols::Endpoint, DistributedRuntime, Runtime};

use crate::EngineConfig;

pub async fn run(
    runtime: Runtime,
    path: String,
    engine_config: EngineConfig,
) -> anyhow::Result<()> {
    // This will attempt to connect to NATS and etcd
    let distributed = DistributedRuntime::from_settings(runtime.clone()).await?;

    let cancel_token = runtime.primary_token().clone();
    let endpoint: Endpoint = path.parse()?;

    let etcd_client = distributed.etcd_client();

    let (ingress, service_name) = match engine_config {
        EngineConfig::StaticFull {
            service_name,
            engine,
        } => (Ingress::for_engine(engine)?, service_name),
        EngineConfig::StaticCore {
            service_name,
            engine: inner_engine,
            card,
        } => {
            let frontend = SegmentSource::<
                SingleIn<NvCreateChatCompletionRequest>,
                ManyOut<Annotated<ChatCompletionResponseDelta>>,
            >::new();
            let preprocessor = OpenAIPreprocessor::new(*card.clone())
                .await?
                .into_operator();
            let backend = Backend::from_mdc(*card.clone()).await?.into_operator();
            let engine = ServiceBackend::from_engine(inner_engine);

            let pipeline = frontend
                .link(preprocessor.forward_edge())?
                .link(backend.forward_edge())?
                .link(engine)?
                .link(backend.backward_edge())?
                .link(preprocessor.backward_edge())?
                .link(frontend)?;

            (Ingress::for_pipeline(pipeline)?, service_name)
        }
        EngineConfig::Dynamic(_) => {
            anyhow::bail!("Cannot use endpoint for both in and out");
        }
    };

    let model_registration = ModelEntry {
        name: service_name.to_string(),
        endpoint: endpoint.clone(),
        model_type: ModelType::Chat,
    };
    etcd_client
        .kv_create(
            path.clone(),
            serde_json::to_vec_pretty(&model_registration)?,
            None,
        )
        .await?;

    let rt_fut = distributed
        .namespace(endpoint.namespace)?
        .component(endpoint.component)?
        .service_builder()
        .create()
        .await?
        .endpoint(endpoint.name)
        .endpoint_builder()
        .handler(ingress)
        .start();

    tokio::select! {
        _ = rt_fut => {
            tracing::debug!("Endpoint ingress ended");
        }
        _ = cancel_token.cancelled() => {
        }
    }
    Ok(())
}
