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

use crate::Result;
use async_trait::async_trait;
use serde::{de::DeserializeOwned, Serialize};
use serde_json::Value;
use std::any::Any;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::Arc;

/// Specification of an action to be performed
pub struct Action {
    pub action: String,
    pub params: Value,
}

/// Trait for executing actions
#[async_trait]
pub trait ActionExecutorBase: Send + Sync {
    async fn execute(&self, params: &(dyn Any + Send + Sync)) -> Result<()>;
}

#[async_trait]
pub trait ExecuteAction<T> {
    async fn execute(&self, params: &T) -> Result<()>;
}

///
pub struct TypedActionExecutor<T, E> {
    executor: E,
    _phantom: PhantomData<T>,
}

impl<T, E> TypedActionExecutor<T, E> {
    pub fn new(executor: E) -> Self {
        Self {
            executor,
            _phantom: PhantomData,
        }
    }
}

impl<T, E> From<E> for TypedActionExecutor<T, E> {
    fn from(executor: E) -> Self {
        Self::new(executor)
    }
}

#[async_trait]
impl<T, E> ActionExecutorBase for TypedActionExecutor<T, E>
where
    T: DeserializeOwned + Send + Sync + 'static,
    E: ExecuteAction<T> + Send + Sync,
{
    async fn execute(&self, params: &(dyn Any + Send + Sync)) -> Result<()> {
        if let Some(typed_params) = params.downcast_ref::<T>() {
            self.executor.execute(typed_params).await
        } else if let Some(value) = params.downcast_ref::<Value>() {
            let typed_params = serde_json::from_value(value.clone())?;
            self.executor.execute(&typed_params).await
        } else {
            Err(anyhow::anyhow!("Invalid parameter type").into())
        }
    }
}

pub struct ActionRegistry {
    executors: HashMap<String, Arc<dyn ActionExecutorBase>>,
}

impl Default for ActionRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ActionRegistry {
    pub fn new() -> Self {
        Self {
            executors: HashMap::new(),
        }
    }

    pub fn register(&mut self, name: impl Into<String>, executor: Arc<dyn ActionExecutorBase>) {
        self.executors.insert(name.into(), executor);
    }

    pub async fn execute_action(&self, action: Action) -> Result<()> {
        let executor = self.executors.get(&action.action).ok_or_else(|| {
            anyhow::anyhow!("No executor registered for action type: {}", action.action)
        })?;

        executor.execute(&action.params).await
    }

    /// Register an executor for a specific type
    ///
    /// # Example
    /// ```rust
    /// use triton_distributed::{actions::{ActionRegistry, ExecuteAction}, Result};
    /// use serde::Deserialize;
    /// use async_trait::async_trait;
    ///
    /// #[derive(Deserialize)]
    /// struct MyParams {
    ///     value: String,
    /// }
    ///
    /// struct MyExecutor;
    ///
    /// #[async_trait]
    /// impl ExecuteAction<MyParams> for MyExecutor {
    ///     async fn execute(&self, params: &MyParams) -> Result<()> {
    ///         println!("Executing with value: {}", params.value);
    ///         Ok(())
    ///     }
    /// }
    ///
    /// let mut registry = ActionRegistry::new();
    /// registry.register_typed::<MyParams, _>("my_action", MyExecutor);
    /// ```
    pub fn register_typed<T, E>(&mut self, name: impl Into<String>, executor: E)
    where
        T: DeserializeOwned + Send + Sync + 'static,
        E: ExecuteAction<T> + Send + Sync + 'static,
    {
        self.register(name, Arc::new(TypedActionExecutor::from(executor)));
    }

    pub async fn execute<T: Serialize>(
        &self,
        action_type: impl Into<String>,
        params: T,
    ) -> Result<()> {
        let params = serde_json::to_value(params)?;
        let action = Action {
            action: action_type.into(),
            params,
        };
        self.execute_action(action).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};
    use std::sync::Mutex;

    #[derive(Debug, Serialize, Deserialize)]
    struct TestParams {
        name: String,
    }

    struct TestExecutor {
        calls: Arc<Mutex<Vec<String>>>,
    }

    #[async_trait]
    impl ExecuteAction<TestParams> for TestExecutor {
        async fn execute(&self, params: &TestParams) -> Result<()> {
            self.calls.lock().unwrap().push(params.name.clone());
            Ok(())
        }
    }

    #[tokio::test]
    async fn test_action_registry() {
        let mut registry = ActionRegistry::new();

        let calls = Arc::new(Mutex::new(Vec::new()));
        let calls_clone = calls.clone();

        // Using the new convenience method
        let executor = TestExecutor { calls };
        registry.register_typed::<TestParams, _>("test", executor);

        // Test with direct type
        let params = TestParams {
            name: "direct_call".into(),
        };
        registry.execute("test", params).await.unwrap();

        // Test with JSON Value
        let action = Action {
            action: "test".into(),
            params: serde_json::json!({
                "name": "json_call"
            }),
        };
        registry.execute_action(action).await.unwrap();

        // Get the executor back through downcasting
        let calls = calls_clone.lock().unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0], "direct_call");
        assert_eq!(calls[1], "json_call");
    }

    #[tokio::test]
    async fn test_enum_action_registry() {
        #[derive(Debug, Clone, Serialize, Deserialize)]
        enum TestAction {
            Add { value: i32 },
            Remove { id: String },
        }

        struct TestEnumExecutor {
            adds: Arc<Mutex<Vec<i32>>>,
            removes: Arc<Mutex<Vec<String>>>,
        }

        #[async_trait]
        impl ExecuteAction<TestAction> for TestEnumExecutor {
            async fn execute(&self, action: &TestAction) -> Result<()> {
                match action {
                    TestAction::Add { value } => {
                        self.adds.lock().unwrap().push(*value);
                    }
                    TestAction::Remove { id } => {
                        self.removes.lock().unwrap().push(id.clone());
                    }
                }
                Ok(())
            }
        }

        let mut registry = ActionRegistry::new();

        let adds = Arc::new(Mutex::new(Vec::new()));
        let removes = Arc::new(Mutex::new(Vec::new()));
        let adds_clone = adds.clone();
        let removes_clone = removes.clone();

        let executor = TestEnumExecutor { adds, removes };
        registry.register_typed::<TestAction, _>("test", executor);

        // Test direct enum variant
        registry
            .execute("test", TestAction::Add { value: 42 })
            .await
            .unwrap();

        // Test with JSON Value
        let action = Action {
            action: "test".into(),
            params: serde_json::json!({
                "Remove": {
                    "id": "test-id"
                }
            }),
        };
        registry.execute_action(action).await.unwrap();

        let adds = adds_clone.lock().unwrap();
        let removes = removes_clone.lock().unwrap();
        assert_eq!(adds.len(), 1);
        assert_eq!(adds[0], 42);
        assert_eq!(removes.len(), 1);
        assert_eq!(removes[0], "test-id");
    }
}
