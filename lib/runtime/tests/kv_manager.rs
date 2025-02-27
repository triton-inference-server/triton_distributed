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

//! Prototype KV Manager
//!
//! The KV Manager will be linked to three components:
//! - ForwardPassTask / Scheduler
//!   - On each forward pass, any slot that has completed a block will:
//!     - Add the block to the Persistence Engine
//!     - Acquire a new block to continue generating
//! - Persistence Engine
//!   - Will perform copies from GPU memory to CPU memory and possibly CPU memory
//!     to some global flash storage
//! - Prefill Descriptor Manager
//!   - New request that require prefill offload, will acquire leases on any shared
//!     blocks and any "net new" blocks that need to be populated from the prefill
//!     instance.
//!

use async_trait::async_trait;
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::{Mutex, Notify};
use triton_distributed_runtime::utils::pool::{
    PoolExt, PoolItem, PoolValue, Returnable, SharedPoolItem,
};

pub struct Block {
    id: u64,
    block_hash: u64,
    sequence_hash: u128,
    position_in_sequence: u32,
}

impl Returnable for Block {}

/// Simple pool of free blocks, later we will make this sequence aware
pub struct FreePool {
    blocks: Arc<Mutex<VecDeque<Block>>>,
}

impl FreePool {
    pub fn new() -> Self {
        Self {
            blocks: Arc::new(Mutex::new(VecDeque::new())),
        }
    }
}

// pub struct Sequence {
//     tokens: Vec<u32>,
//     shared_blocks: Vec<SharedPoolItem<Block>>,
//     current_block: PoolItem<Block>,
// }
