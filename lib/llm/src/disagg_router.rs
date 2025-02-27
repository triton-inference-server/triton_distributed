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

#[derive(Clone)]
pub struct DisaggregatedRouter {
    max_local_prefill_length: i32,
}

impl DisaggregatedRouter {
    pub fn new(max_local_prefill_length: i32) -> Self {
        DisaggregatedRouter { max_local_prefill_length }
    }

    pub fn prefill_remote(&self, prefill_length: i32, prefix_hit_length: i32) -> bool {
        // schedule the request purely based on the prefill length
        // TODO: apply math models and compare local vs remote prefill TTFT
        return prefill_length - prefix_hit_length > self.max_local_prefill_length
    }

    pub fn update_value(&mut self, max_local_prefill_length: i32) {
        self.max_local_prefill_length = max_local_prefill_length;
    }

}