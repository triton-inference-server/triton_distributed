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
    Pool, PoolExt, PoolItem, PoolValue, Returnable, SharedPoolItem,
};

pub struct Block {
    id: u64,
    block_hash: u64,
    sequence_hash: u128,
    position_in_sequence: u32,
}

impl Returnable for Block {}

pub type UniqueBlock = PoolItem<Block, Pool<Block>>;
pub type SharedBlock = SharedPoolItem<Block, Pool<Block>>;

/// A wrapper around a time-critical item that will determine the amount of elapsed/walltime
/// since the item was created. The `deadline` is optional and if not set, the item will be
/// considered to have no time constraints. If the `deadline` is set, the item will be will
/// increment a [prometheus::Counter] if the deadline is exceeded.
///
/// In this manner, we can monitor the time-criticality of the item and take action if it is
/// taking too long to process.
// pub struct TimeCritical<T> {
//     // pub timestamp: Instant,
//     // pub item: T,
//     // pub deadline: Option<Instant>,
// }

pub struct Sequence {
    tokens: Vec<u32>,
    shared_blocks: Vec<SharedBlock>,
    current_block: UniqueBlock,
}

/// Adapt the KvIndexer to hold Block information
pub struct DeviceRadixTree {}

/// Adapt the KvIndexer to hold Block information
pub struct HostRadixTree {}

/// Owner of the radix trees and the block pool
pub struct KvBlockManager {}

/// The [Scheduler] is responsible for determining which [Sequence] objects should be
/// scheduled for the next forward pass.
///
/// The [Scheduler] will prepare a [Sequence] object for each request and pass that [Sequence]
/// to either the [ForwardPassEngine] or the [PrefillHandler] depending the size of the
/// ISL and "net-new" tokens that need to be prefilled to the [Sequence].
///
/// The [Scheduler] has have multiple [Sequences][Sequence] offloaded to the [PrefillHandler];
/// however, some care needs to be taken that that value is not "too large" as the blocks
/// held by the [Sequence] can not be reused or repurposed by eviction.
pub struct Scheduler {
    // slots: BTreeMap<u64, Sequence>,
    // pending: VecDeque<Sequencd>,
}

/// The [ForwardPassEngine] is responsible for scheduling the forward pass of the model.
/// It will receive requests from the scheduler that will have the set of SharedBlocks that
/// associated with the current request tied to a Sequence object.
///
/// The [ForwardPassEngine] appends new tokens to the current block of the [Sequence]. When
/// the current block is full, it is converted to an immutable [SharedBlock] and a copy/clone
/// is passed to the [PersistenceEngine] via an mpsc::Sender<TimeCritical<SharedBlock>>.
///
/// The [ForwardPassEngine] should spawn async tasks per forward pass to evaluate the potential
/// of each [Sequence] and determine how many blocks it could return to the [FreePool] if it was
/// evicted.
///
/// We only want to evict a [Sequence] if it can free enough blocks to be worth the overhead of
/// evicting it and most critically, that we have persisted all evicted blocks in host memory.
/// This will avoid the need to re-prefill the blocks when the sequence is rescheduled.
///
/// The [ForwardPassEngine] should also evaluate the potential of each [Sequence] to be
/// prefilled and if so, it will return a [PrefillHandler] to the caller.
pub struct ForwardPassEngine {
    // scheduler: Scheduler,
    // kv_manager: KvBlockManager,
    // persistence_engine: PersistenceEngine,
}

/// The [PersistenceEngine] is responsible for copying blocks from GPU memory to
/// to either host memory or some form of persistent storage.
///
/// The [PersistenceEngine] will have a mpsc receiver of SharedBlock. Each block can
/// be handled independently and freed after the copy is complete.
///
/// We must time each SharedBlock as it enters the channel, so perhaps we wrap the incoming
/// SharedBlock in a timestamped context.
///
/// Holding SharedBlocks forbids their reuse, so we need to carefully and accurately monitor
/// the state of this engine so it is not starving the ForwardPass [Scheduler] of free blocks.
pub struct PersistenceEngine {}

/// The [PrefillHandler] is responsible for acquiring blocks from the [KvBlockManager] for a
/// given request. The input sequence length will be evaluated and two sets of blocks will be
/// returned to the caller:
///   - Vec<SharedBlock>
///   - Vec<UniqueBlock>
///
/// The `Vec<SharedBlock>` are the blocks that matched inflight radix tree. By acquiring a
/// [SharedBlock], this ensure that the blocks cannot be returned to the [FreePool].
///
/// The `Vec<UniqueBlock>` are the new blocks that are not present in the inflight radix tree
/// which need to be prefilled. The decision to prefill locally via chunking of to offload to
/// dedicated prefill workers can be made once the target destinations for the KV are determined.
pub struct PrefillHandler {}

/// The [MigrationEngine] is responsible for migrating blocks from one physical machine to another.
/// In an ideal world, this transfer is over NVLink or ConnectX InfiniBand; however, any reasonably
/// fast transfer will suffice.
///
/// The [MigrationEngine] spawns tasks that operate in two paradigms:
/// - RDMA Passive Source: The task will acquire [SharedBlocks][SharedBlock] from the [KvBlockManager]
///   and hold them until a RDMA GET COMPLETION notification is received. Essentially, the task which
///   holds the [SharedBlocks][SharedBlock] is simply responsible for ensuring the memory is pinned
///   and not returned to the [FreePool] over the duration of the RDMA GET.
/// - RDMA Active Puller: The task will receive a set of [SharedBlocks][SharedBlock]. The block list
///   is a set of block_ids and a remote target. The task will initiate the RDMA GETs via the NIXL
///   library and then wait for completion. Upon completion, and event or active message event will
///   be triggered on each RDMA Passive Source to trigger task completion and resource dereferencing.
///
pub struct MigrationEngine {}
