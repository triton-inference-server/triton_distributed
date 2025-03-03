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

pub mod reserved;
pub mod reuse;
pub mod sequence;

use reserved::*;
use reuse::*;

use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    sync::{
        atomic::{AtomicU64, AtomicU8},
        Arc, RwLock,
    },
};

use async_trait::async_trait;
use derive_getters::Dissolve;
use tokio::time::Instant;
use triton_distributed_runtime::{
    raise,
    utils::pool::{Pool, PoolExt, PoolItem, PoolValue, Returnable, SharedPoolItem},
    Result,
};

use crate::tokens::{PartialTokenBlock, SequenceHash, TokenBlock, TokenSequence, Tokens};

use tracing as log;

pub trait Storage {}

pub type UniqueBlock = PoolItem<KvBlock>;
pub type SharedBlock = SharedPoolItem<KvBlock>;
pub enum StorageState {
    Present = 0,
    Pending = 1,
    Absent = 2,
}

#[derive(Default)]
pub struct KvBlock {
    pub token_block: TokenBlock,
    // pub device_state: Arc<AtomicU8>,
    // pub host_state: Arc<AtomicU8>,
    // pub storage_state: Arc<AtomicU8>,
    pub priority: u32,
    pub return_tick: u64,
}

impl StorageState {
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(StorageState::Present),
            1 => Some(StorageState::Pending),
            2 => Some(StorageState::Absent),
            _ => None,
        }
    }

    pub fn as_u8(&self) -> u8 {
        match self {
            StorageState::Present => 0,
            StorageState::Pending => 1,
            StorageState::Absent => 2,
        }
    }
}

impl KvBlock {
    pub fn update_token_block(&mut self, token_block: TokenBlock) {
        self.token_block = token_block;
    }

    pub(crate) fn reset(&mut self) {
        self.token_block = TokenBlock::default();
        self.priority = 0;
        self.return_tick = 0;
    }
}

impl Returnable for KvBlock {
    fn on_return(&mut self) {}
}

// pub struct KvBlockManager {
//     available_blocks: AvailableBlocks,
//     inflight_blocks: ReservedBlocks,
//     block_size: usize,
// }

// impl KvBlockManager {
//     pub async fn new(block_size: usize) -> Self {
//         Self {
//             available_blocks: AvailableBlocks::new().await,
//             inflight_blocks: ReservedBlocks::new(block_size),
//             block_size,
//         }
//     }

//     pub fn prepare_prefill_sequence(&mut self, tokens: Tokens) -> Result<PrefillMatched> {
//         log::debug!("adding request with {} tokens", tokens.len());

//         let seq = TokenSequence::new(tokens, self.block_size);
//         let (blocks, tail_block) = seq.into_parts();
//         log::debug!("request translates to {} blocks", blocks.len());

//         // first match blocks to inflight blocks
//         let mut inflight_blocks = self.inflight_blocks.match_token_blocks(&blocks)?;
//         log::debug!("matched {} inflight blocks", inflight_blocks.len());

//         // shift the blocks to the left by the number of inflight blocks
//         let unmatched_blocks = &blocks[inflight_blocks.len()..];

//         // match the remaining blocks to freed gpu blocks (available_blocks)
//         let unregistered_blocks = self.available_blocks.match_token_blocks(unmatched_blocks);
//         log::debug!("matched {} freed blocks", unregistered_blocks.len());

//         // the blocks from the freed blocks pool must be registered as inflight blocks
//         // todo - we might have to register the list of unregistered blocks as a single transaction
//         for block in unregistered_blocks {
//             inflight_blocks.push(self.inflight_blocks.register(block)?);
//         }

//         // the remaining blocks are the unmatched blocks
//         let remaining_blocks = blocks.into_iter().skip(inflight_blocks.len()).collect();

//         Ok(PrefillMatched {
//             inflight_blocks,
//             remaining_blocks,
//             tail_block,
//         })
//     }
// }

/// State of the prefill sequence that is not inflight
pub struct PrefillPending {
    /// Complete blocks that are not inflight
    remaining_blocks: Vec<TokenBlock>,

    /// Tokens that do not form a complete block
    remaining_tokens: Tokens,
}

/// The [UniqueBlocks][UniqueBlock] have been updated with the token sequences for each block.
/// The KV Blocks are allocated and in a mutable state awaiting for the prefill operation to complete.
/// Upon completion, each "complete" [UniqueBlock] will be registered and converted to an [ReservedBlock].
/// If the last [UniqueBlock] is not complete, it will be passed to the Sequence as the current block
/// remaining in the form of a [UniqueBlock].
pub struct PrefillScheduled {
    unique_blocks: Vec<UniqueBlock>,
}

pub enum PrefillState {
    /// Initialized with only the input sequence tokens
    Initial(Tokens),

    /// The prefill is pending and has not been scheduled for execution
    Pending(PrefillPending),

    /// The prefill is scheduled for local execution
    Scheduled(PrefillScheduled),

    /// The prefill is scheduled for remote execution
    Offloaded(PrefillScheduled),

    /// The prefill is complete and the sequence is in generation mode
    Complete,
}

impl PrefillState {
    pub fn init(tokens: Tokens) -> Self {
        Self::Initial(tokens)
    }

    pub fn from_blocks(blocks: Vec<TokenBlock>, remainding_tokens: Option<Tokens>) -> Self {
        let remaining_tokens = remainding_tokens.unwrap_or_default();
        if blocks.is_empty() && remaining_tokens.is_empty() {
            Self::Complete
        } else {
            Self::Pending(PrefillPending {
                remaining_blocks: blocks,
                remaining_tokens,
            })
        }
    }
}

struct PrefillInit(Tokens);

#[derive(Dissolve)]
struct PrefillMatched {
    inflight_blocks: Vec<ReservedBlock>,
    remaining_blocks: Vec<TokenBlock>,
    tail_block: PartialTokenBlock,
}

/// When we issue offloaded prefill, we now have all the memory reserved for the rdma puts
/// we simply need to await the completion of all the puts
/// we need to prepare the rdma descriptors for each block
/// since we will have shared descriptors available via etcd, we only need to provide the
/// proper index for each block
///
/// the task driving this should be the async task processing the request before scheduling
/// if the offload fails it should capture that.
struct PrefillOffload {
    inflight_blocks: Vec<ReservedBlock>,
    prefill_blocks: Vec<UniqueBlock>,
    tail_block: UniqueBlock,
}

// impl PrefillOffload {
//     /// when this function completes, the prefill blocks have their kv data populated
//     /// we need to register the blocks and convert them to inflight blocks
//     /// then add to scheduler for generation
//     async fn execute(&self) -> Result<()> {}
// }

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use triton_distributed_runtime::logging::init;

//     #[tokio::test]
//     async fn test() {
//         init();

//         let mut manager = KvBlockManager::new(2);

//         for _ in 0..100 {
//             manager.available_blocks.insert(KvBlock::default());
//         }

//         let tokens = Tokens::from([0_i32, 1, 2, 3, 4, 5, 6, 7, 8].as_ref());

//         // this is good for the scheduler to make a local decision as it now knows how many
//         // net-new blocks need to be prefilled
//         let sequence = manager.prepare_prefill_sequence(tokens).unwrap();

//         assert_eq!(sequence.inflight_blocks.len(), 0);
//         assert_eq!(sequence.remaining_blocks.len(), 4);
//     }
// }
