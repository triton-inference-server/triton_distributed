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

use super::*;

#[derive(Debug, Clone)]
pub struct Function {
    component: Component,

    // todo - restrict alphabet
    /// Endpoint name
    name: String,
}

impl Function {
    pub fn new(component: Component, name: String) -> Result<Self> {
        validate_allowed_chars(&name)?;
        Ok(Self { component, name })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn component(&self) -> &Component {
        &self.component
    }

    pub fn path(&self) -> String {
        format!("{}/{}", self.component.path(), self.name)
    }

    pub fn etcd_path(&self) -> String {
        format!("{}/{}", self.component.etcd_path(), self.name)
    }

    pub fn etcd_path_with_id(&self, lease_id: i64) -> String {
        format!("{}:{:x}", self.etcd_path(), lease_id)
    }

    pub fn name_with_id(&self, lease_id: i64) -> String {
        format!("{}-{:x}", self.name, lease_id)
    }

    pub fn subject(&self) -> String {
        format!("{}.{}", self.component.service_name(), self.name)
    }

    /// Subject to an instance of the [Endpoint] with a specific lease id
    pub fn subject_to(&self, lease_id: i64) -> String {
        format!(
            "{}.{}",
            self.component.service_name(),
            self.name_with_id(lease_id)
        )
    }

    pub async fn client<Req, Resp>(&self) -> Result<client::Client<Req, Resp>>
    where
        Req: Serialize + Send + Sync + 'static,
        Resp: for<'de> Deserialize<'de> + Send + Sync + 'static,
    {
        client::Client::new(self.clone()).await
    }

    pub fn endpoint_builder(&self) -> endpoint::EndpointConfigBuilder {
        endpoint::EndpointConfigBuilder::from_endpoint(self.clone())
    }
}

impl DistributedRuntimeProvider for Function {
    fn drt(&self) -> &DistributedRuntime {
        self.component.drt()
    }
}

impl RuntimeProvider for Function {
    fn rt(&self) -> &Runtime {
        self.component.rt()
    }
}
