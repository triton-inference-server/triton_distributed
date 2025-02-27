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

use serde::{Deserialize, Serialize};

pub const DEFAULT_NAMESPACE: &str = "triton-init";

#[derive(Debug, Clone, Serialize, Deserialize)]
// Dummy Stats object to demonstrate how to attach a custom stats handler
pub struct MyStats {
    pub requests_active_slots: u32,
    pub requests_total_slots: u32,
    pub kv_blocks_active: u32,
    pub kv_blocks_total: u32,
}
