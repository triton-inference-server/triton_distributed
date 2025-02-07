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

use async_nats::service::{endpoint, Service};

pub type StatsHandler =
    Box<dyn FnMut(String, endpoint::Stats) -> serde_json::Value + Send + Sync + 'static>;

#[derive(Educe, Builder, Dissolve)]
#[educe(Debug)]
#[builder(pattern = "owned", build_fn(private, name = "build_internal"))]
pub struct ServiceConfig {
    #[builder(private)]
    component: Component,

    /// Description
    #[builder(default)]
    description: Option<String>,

    /// Lease
    #[educe(Debug(ignore))]
    #[builder(default)]
    lease: Option<Lease>,

    // todo - make optional - if None, then skip making the endpoint
    // and skip making the service-endpoint discoverable.
    /// Endpoint handler
    #[educe(Debug(ignore))]
    #[builder(default)]
    stats_handler: Option<StatsHandler>,
}

impl ServiceConfigBuilder {
    /// Create the [`Component`]'s service and store it in the registry.
    pub async fn create(self) -> Result<Component> {
        let version = "0.0.1".to_string();

        // dissolve the config
        let (component, description, lease, stat_handler) = self.build_internal()?.dissolve();

        // either use a custom lease or use the default lease
        let lease = lease.unwrap_or(component.drt.primary_lease());

        let service_name = component.slug();
        let description = description.unwrap_or(format!(
            "Triton Component {} in {}",
            component.name, component.namespace
        ));

        let mut guard = component.drt.component_registry.services.lock().await;

        if guard.contains_key(&component.etcd_path()) {
            return Err(anyhow::anyhow!("Service already exists"));
        }

        // create service on the secondary runtime
        let secondary = component.drt.runtime.secondary.clone();
        let builder = component.drt.nats_client.client().service_builder();
        let service = secondary
            .spawn(async move {
                // unwrap the stats handler
                let builder = match stat_handler {
                    Some(handler) => builder.stats_handler(handler),
                    None => builder,
                };

                tracing::debug!("Starting service: {}", service_name);

                builder
                    .description(description)
                    .start(service_name.to_string(), version)
                    .await
            })
            .await?
            .map_err(|e| anyhow::anyhow!("Failed to start service: {e}"))?;

        guard.insert(component.etcd_path(), service);
        drop(guard);

        Ok(component)
    }
}

impl ServiceConfigBuilder {
    pub(crate) fn from_component(component: Component) -> Self {
        Self::default().component(component)
    }
}

// // Wrap the optional user callback method in a closure that appends the lease_id to the response
// fn wrap_callback(
//     callback: Option<Box<dyn FnMut(String, Stats) -> Value + Send + Sync>>,
//     lease_id: i64,
// ) -> Box<dyn FnMut(String, Stats) -> Value + Send + Sync> {
//     let callback = Arc::new(Mutex::new(callback)); // Wrap in Arc<Mutex> for shared access

//     Box::new(move |subject: String, stats: Stats| -> Value {
//         let mut callback_lock = callback.lock().unwrap();
//         if let Some(cb) = callback_lock.as_mut() {
//             let mut result = cb(subject, stats); // Call the user-defined callback
//             if let Some(obj) = result.as_object_mut() {
//                 obj.insert("lease_id".to_string(), json!(lease_id)); // Append lease_id
//             }
//             result
//         } else {
//             json!({ "error": "callback not set", "lease_id": lease_id }) // Default response
//         }
//     })
// }