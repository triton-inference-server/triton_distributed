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

use derive_getters::Dissolve;

use super::*;

#[derive(Educe, Builder, Dissolve)]
#[educe(Debug)]
#[builder(pattern = "owned", build_fn(private, name = "build_internal"))]
pub struct EndpointConfig {
    #[builder(private)]
    endpoint: Endpoint,

    /// Lease
    #[educe(Debug(ignore))]
    #[builder(default)]
    lease: Option<Lease>,

    /// Endpoint handler
    #[educe(Debug(ignore))]
    handler: Arc<dyn PushWorkHandler>,
}

impl EndpointConfigBuilder {
    pub(crate) fn from_endpoint(endpoint: Endpoint) -> Self {
        Self::default().endpoint(endpoint)
    }

    pub async fn start(self) -> Result<()> {
        let (endpoint, lease, handler) = self.build_internal()?.dissolve();
        let lease = lease.unwrap_or(endpoint.component.drt.lease());

        tracing::debug!(
            "Starting endpoint: {}",
            endpoint.etcd_path_with_id(lease.id())
        );

        let group = endpoint
            .component
            .drt
            .component_registry
            .services
            .lock()
            .await
            .get(&endpoint.component.etcd_path())
            .map(|service| service.group(endpoint.component.slug()))
            .ok_or(error!("Service not found"))?;

        // let group = service.group(service_name.as_str());

        // creates an endpoint for the service
        let service_endpoint = group
            .endpoint(&endpoint.name_with_id(lease.id()))
            .await
            .map_err(|e| anyhow::anyhow!("Failed to start endpoint: {e}"))?;

        let cancel_token = lease.child_token();

        let push_endpoint = PushEndpoint::builder()
            .service_handler(handler)
            .cancellation_token(cancel_token.clone())
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to build push endpoint: {e}"))?;

        // launch!
        let task = tokio::spawn(push_endpoint.start(service_endpoint));

        // tracing::debug!(worker_id, "endpoint subject: {}", subject);

        // make the components service endpoint discovery in etcd

        // client.register_service()
        let info = ComponentEndpointInfo {
            component: endpoint.component.name.clone(),
            endpoint: endpoint.name.clone(),
            namespace: endpoint.component.namespace.clone(),
            lease_id: lease.id(),
            transport: TransportType::NatsTcp(endpoint.subject(lease.id())),
        };

        let info = serde_json::to_vec_pretty(&info)?;

        if let Err(e) = endpoint
            .component
            .drt
            .etcd_client
            .kv_create(
                endpoint.etcd_path_with_id(lease.id()),
                info,
                Some(lease.id()),
            )
            .await
        {
            tracing::error!("Failed to register discoverable service: {:?}", e);
            cancel_token.cancel();
            return Err(error!("Failed to register discoverable service"));
        }

        task.await??;

        Ok(())
    }
}
