// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

use serde::{Deserialize, Serialize};

use super::FinishReason;
use crate::protocols::TokenIdType;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct PostprocessedResponse {
    /// Model Deployment Card checksum
    pub mdcsum: String,

    // if the number of slots for a given request is greater than 1
    // this indicates the index of the slot for the response
    pub index: Option<usize>,

    pub finish_reason: Option<FinishReason>,

    // new token_ids
    pub token_ids: Vec<TokenIdType>,

    // tokens
    pub tokens: Option<Vec<Option<String>>>,

    // decoded text
    pub text: Option<String>,

    /// cumulative log probabilities
    pub cum_log_probs: Option<f64>,
}
