use super::*;

type ReservedBlockMap = Arc<RwLock<HashMap<SequenceHash, Arc<ReservedBlockInner>>>>;

#[derive(Clone)]
pub struct ReservedBlock {
    inner: Arc<ReservedBlockInner>,
}

impl ReservedBlock {
    fn new(inner: Arc<ReservedBlockInner>) -> Self {
        Self { inner }
    }

    pub fn inflight_count(&self) -> usize {
        // the inflight map holds an copy of the inner, so we subtract one
        Arc::strong_count(&self.inner) - 1
    }
}

impl std::ops::Deref for ReservedBlock {
    type Target = SharedBlock;

    fn deref(&self) -> &Self::Target {
        &self.inner.block
    }
}

struct ReservedBlockInner {
    block: SharedBlock,
    map: ReservedBlockMap,
}

impl Drop for ReservedBlockInner {
    fn drop(&mut self) {
        let mut map = self.map.write().unwrap();
        map.remove(&self.block.token_block.sequence_hash());
    }
}

/// [ReservedBlocks] is a collection of inflight blocks that are actively being used
pub struct ReservedBlocks {
    block_size: usize,
    blocks: ReservedBlockMap,
}

impl ReservedBlocks {
    pub fn new(block_size: usize) -> Self {
        Self {
            block_size,
            blocks: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub fn match_sequence_hashes(
        &self,
        sequence_hashes: &[SequenceHash],
    ) -> Result<Vec<ReservedBlock>> {
        let mut inflight_blocks = Vec::new();
        let map = self.blocks.read().unwrap();
        for sequence_hash in sequence_hashes {
            if let Some(inner) = map.get(sequence_hash) {
                inflight_blocks.push(ReservedBlock::new(inner.clone()));
            } else {
                break;
            }
        }
        Ok(inflight_blocks)
    }

    /// Match the list of blocks to inflight blocks
    ///
    /// This will return a [Vec<ReservedBlock>] that match the sequence hashes
    /// in the order of the token blocks.
    ///
    /// The matching is done in order, with the first block in the list being the first
    /// block in the token blocks list.
    ///
    /// If a block is not found, the function will return the list of matched blocks
    /// and the remaining blocks will not be included.
    pub fn match_token_blocks(&self, token_blocks: &[TokenBlock]) -> Result<Vec<ReservedBlock>> {
        let mut inflight_blocks = Vec::new();
        let map = self.blocks.read().unwrap();
        for token_block in token_blocks {
            let sequence_hash = token_block.sequence_hash();
            if let Some(inner) = map.get(&sequence_hash) {
                inflight_blocks.push(ReservedBlock::new(inner.clone()));
            } else {
                break;
            }
        }
        Ok(inflight_blocks)
    }

    pub fn register(&mut self, block: UniqueBlock) -> Result<ReservedBlock> {
        let sequence_hash = block.token_block.sequence_hash();
        let shared = block.into_shared();

        if shared.token_block.tokens().len() != self.block_size {
            raise!("Block size mismatch");
        }

        // if the block already exists, we drop the block the user passed in and return the existing block
        // this should return the passed in block to the free pool
        let mut map = self.blocks.write().unwrap();
        if let Some(existing_block) = map.get(&sequence_hash) {
            // return an ReservedBlock with the existing block
            // the passed in block will be dropped and returned to the pool
            // this could happen if two sequences are building the same block at the same time,
            // the first sequence to finish and register the block will insert it into the map
            return Ok(ReservedBlock::new(existing_block.clone()));
        } else {
            // Insert the new block and create an ReservedBlock from it
            let inner = Arc::new(ReservedBlockInner {
                block: shared,
                map: self.blocks.clone(),
            });

            map.insert(sequence_hash, inner.clone());

            return Ok(ReservedBlock::new(inner));
        }
    }
}
