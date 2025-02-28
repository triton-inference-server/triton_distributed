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

use async_once_cell::OnceCell as AsyncOnceCell;
use std::sync::atomic::{AtomicU32, Ordering};

// need a different file cause of this import
use triton_distributed_runtime::{DistributedRuntime, Worker};
use triton_distributed_llm::kv_router::{
    indexer::compute_block_hash_for_seq, protocols::*, publisher::KvEventPublisher,
};

static WARN_COUNT: AtomicU32 = AtomicU32::new(0);
static WK: OnceCell<Worker> = OnceCell::new();
static DRT: AsyncOnceCell<DistributedRuntime> = AsyncOnceCell::new();
static KV_PUB: OnceCell<KvEventPublisher> = OnceCell::new();


#[pyfunction]
pub fn triton_llm_event_init(
    namespace: String,
    component: String,
    worker_id: i64,
) -> TritonLlmResult {
    let wk = match WK.get_or_try_init(Worker::from_settings) {
        Ok(wk) => wk.clone(),
        Err(e) => {
            eprintln!("Failed to initialize runtime: {:?}", e);
            return TritonLlmResult::ERR;
        }
    };

    let rt = wk.runtime();
    let secondary = rt.secondary().clone();
    let result = secondary.block_on(async {
        // Initialize the distributed runtime
        match DRT
            .get_or_try_init(async { DistributedRuntime::from_settings(rt.clone()).await })
            .await
        {
            Ok(_) => Ok(()),
            Err(e) => {
                eprintln!("Failed to initialize distributed runtime: {:?}", e);
                Err(TritonLlmResult::ERR)
            }
        }
    });

    match result {
        Ok(_) => match KV_PUB
            .get_or_try_init(move || triton_create_kv_publisher(namespace, component, worker_id))
        {
            Ok(_) => TritonLlmResult::OK,
            Err(e) => {
                eprintln!("Failed to initialize distributed runtime: {:?}", e);
                TritonLlmResult::ERR
            }
        },
        Err(e) => e,
    }
}


fn triton_create_kv_publisher(
    namespace: String,
    component: String,
    worker_id: i64,
) -> Result<KvEventPublisher, anyhow::Error> {
    tracing::info!("Creating KV Publisher for model: {}", component);
    match DRT
        .get()
        .ok_or(anyhow::Error::msg("Could not get Distributed Runtime"))
    {
        Ok(drt) => {
            let backend = drt.namespace(namespace)?.component(component)?;
            KvEventPublisher::new(drt.clone(), backend, worker_id)
        }
        Err(e) => Err(e),
    }
}

fn kv_event_create_stored_block_from_parts(
    block_hash: u64,
    token_ids: Vec<u32>,
    num_tokens: usize,
    _lora_id: u64,
) -> KvCacheStoredBlockData {
    let tokens_hash =
        compute_block_hash_for_seq(&token_ids)[0];
    KvCacheStoredBlockData {
        block_hash: ExternalSequenceBlockHash(block_hash),
        tokens_hash,
    }
}

fn kv_event_create_stored_from_parts(
    event_id: u64,
    token_ids: Vec<u32>,
    num_block_tokens: Vec<usize>,
    block_ids: Vec<u64>,
    num_blocks: usize,
    parent_hash: Option<u64>,
    lora_id: u64,
) -> KvCacheEvent {
    let mut blocks: Vec<KvCacheStoredBlockData> = Vec::new();

    let mut token_offset: usize = 0;
    for block_idx in 0..num_blocks {
        let block_hash = block_ids[block_idx];
        let tokens = &token_ids[token_offset..token_offset + num_block_tokens[block_idx]];
        let num_toks = num_block_tokens[block_idx];
        // compute hash only apply to full block (KV_BLOCK_SIZE token)
        if num_toks != 64 {
            if WARN_COUNT
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |c| {
                    if c < 3 {
                        Some(c + 1)
                    } else {
                        None
                    }
                })
                .is_ok()
            {
                tracing::warn!(
                    "Block size must be 64 tokens to be published. Block size is: {}",
                    num_toks
                );
            }
            break;
        }
        token_offset += num_toks;
        blocks.push(kv_event_create_stored_block_from_parts(
            block_hash, tokens.to_vec(), num_toks, lora_id,
        ));
    }

    KvCacheEvent {
        data: KvCacheEventData::Stored(KvCacheStoreData {
            blocks,
            parent_hash: parent_hash.map(ExternalSequenceBlockHash),
        }),
        event_id,
    }
}

fn kv_event_create_removed_from_parts(
    event_id: u64,
    block_ids: Vec<u64>,
    num_blocks: usize,
) -> KvCacheEvent {
    let block_hashes: Vec<ExternalSequenceBlockHash> =
        block_ids
            .iter()
            .map(|&v| ExternalSequenceBlockHash(v))
            .collect();
    KvCacheEvent {
        event_id,
        data: KvCacheEventData::Removed(KvCacheRemoveData { block_hashes }),
    }
}

#[pyfunction(signature = (event_id, token_ids, num_block_tokens, block_ids, num_blocks, lora_id, parent_hash = None))]
pub fn triton_kv_event_publish_stored(
    event_id: u64,
    token_ids: Vec<u32>,
    num_block_tokens: Vec<usize>,
    block_ids: Vec<u64>,
    num_blocks: usize,
    lora_id: u64,
    parent_hash: Option<u64>,
) -> TritonLlmResult {
    let publisher = KV_PUB.get().unwrap();
    let parent_hash = match parent_hash {
        Some(h) => Some(h),
        None => None,
    };
    // Some(unsafe { *parent_hash })

    let event = kv_event_create_stored_from_parts(
        event_id,
        token_ids,
        num_block_tokens,
        block_ids,
        num_blocks,
        parent_hash,
        lora_id,
    );
    match publisher.publish(event) {
        Ok(_) => TritonLlmResult::OK,
        Err(e) => {
            eprintln!("Error publishing stored kv event {:?}", e);
            TritonLlmResult::ERR
        }
    }
}

#[pyfunction]
pub fn triton_kv_event_publish_removed(
    event_id: u64,
    block_ids: Vec<u64>,
    num_blocks: usize,
) -> TritonLlmResult {
    let publisher = KV_PUB.get().unwrap();
    let event = kv_event_create_removed_from_parts(event_id, block_ids, num_blocks);
    match publisher.publish(event) {
        Ok(_) => TritonLlmResult::OK,
        Err(e) => {
            eprintln!("Error publishing removed kv event {:?}", e);
            TritonLlmResult::ERR
        }
    }
}