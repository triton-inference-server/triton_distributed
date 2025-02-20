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

//! Runnable Module.
//!
//! This module provides a way to run a task in a runtime.
//!

use std::{
    pin::Pin,
    task::{Context, Poll},
};

pub use crate::{Error, Result};
pub use async_trait::async_trait;
pub use tokio::task::JoinHandle;
pub use tokio_util::sync::CancellationToken;

#[async_trait]
pub trait ExecutionHandle {
    fn is_finished(&self) -> bool;
    fn is_cancelled(&self) -> bool;
    fn cancel(&self);
    fn cancellation_token(&self) -> CancellationToken;
    fn handle(self) -> JoinHandle<Result<()>>;
}
