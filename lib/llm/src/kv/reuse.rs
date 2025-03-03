use tokio::{
    sync::{mpsc, oneshot},
    task::JoinHandle,
};
use triton_distributed_runtime::utils::pool::ReturnHandle;

use super::*;

pub struct AvailableBlocks {
    match_tx: mpsc::UnboundedSender<MatchRequest>,
    return_tx: mpsc::UnboundedSender<PoolValue<KvBlock>>,
    control_tx: mpsc::UnboundedSender<ControlRequest>,
    join_handle: JoinHandle<()>,
    return_handle: Arc<ReturnHandleImpl>,
}

impl AvailableBlocks {
    pub async fn match_blocks(&self, hashes: Vec<SequenceHash>) -> Result<Vec<PoolItem<KvBlock>>> {
        let (tx, rx) = oneshot::channel();
        if self
            .match_tx
            .send(MatchRequest::MatchMultiple(MatchMultiple {
                hashes,
                return_handle: self.return_handle.clone(),
                tx,
            }))
            .is_err()
        {
            raise!("failed to send match request; channel closed");
        }

        let matched_blocks = rx.await?;
        Ok(matched_blocks)
    }

    pub async fn take_blocks(&self, count: u32) -> Result<Vec<PoolItem<KvBlock>>> {
        let (tx, rx) = oneshot::channel();
        if self
            .match_tx
            .send(MatchRequest::Take(Take {
                count,
                return_handle: self.return_handle.clone(),
                tx,
            }))
            .is_err()
        {
            raise!("failed to send take request; channel closed");
        }

        let matched_blocks = rx.await?;
        Ok(matched_blocks)
    }

    // pub async fn reset_all(&self) -> Result<()> {
    //     let (tx, rx) = oneshot::channel();
    //     if self.control_tx.send(ControlRequest::ResetAll).is_err() {
    //         raise!("failed to send reset all request; channel closed");
    //     }

    // }
}

struct ReturnHandleImpl {
    return_tx: mpsc::UnboundedSender<PoolValue<KvBlock>>,
}

impl ReturnHandle<KvBlock> for ReturnHandleImpl {
    fn return_to_pool(&self, value: PoolValue<KvBlock>) {
        if self.return_tx.send(value).is_err() {
            log::trace!("Failed to return block to pool");
        }
    }
}

impl AvailableBlocks {
    pub async fn new() -> Self {
        let (match_tx, match_rx) = mpsc::unbounded_channel();
        let (return_tx, return_rx) = mpsc::unbounded_channel();
        let (control_tx, control_rx) = mpsc::unbounded_channel();

        let return_tx_clone = return_tx.clone();
        let return_handle = Arc::new(ReturnHandleImpl {
            return_tx: return_tx_clone,
        });

        let join_handle = tokio::spawn(progress_engine(match_rx, return_rx, control_rx));

        Self {
            match_tx,
            return_tx,
            control_tx,
            join_handle,
            return_handle,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PriorityKey {
    priority: u32,
    return_tick: u64,
    sequence_hash: SequenceHash,
}

// customize ord and partial ord for to store first by priority (lowest to highest), then by return_tick (lowest to highest)
impl PartialOrd for PriorityKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PriorityKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.priority
            .cmp(&other.priority)
            .then(self.return_tick.cmp(&other.return_tick))
    }
}

impl From<&KvBlock> for PriorityKey {
    fn from(block: &KvBlock) -> Self {
        Self {
            priority: 0,
            return_tick: block.return_tick,
            sequence_hash: block.token_block.sequence_hash(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct DeadlineKey {
    deadline: Instant,
    priority: PriorityKey,
}

#[derive(Default)]
struct AvailableBlocksState {
    // Direct lookup by sequence_hash
    lookup_map: HashMap<SequenceHash, PoolValue<KvBlock>>,

    // // Ordered by timestamp (oldest first)
    priority_set: BTreeMap<PriorityKey, SequenceHash>,

    // Fully Uninitialized
    uninitialized_set: VecDeque<PoolValue<KvBlock>>,

    // Return Tick
    return_tick: u64,

    // Total blocks
    total_blocks: u64,

    // Available blocks
    available_blocks: u64,
}

impl AvailableBlocksState {
    // Insert an item with a given key and sequence_hash
    fn insert(&mut self, block: PoolValue<KvBlock>) {
        let sequence_hash = block.token_block.sequence_hash();
        log::debug!(sequence_hash, "inserting block into available blocks");

        // If we already have an entry for this sequence hash, we need to move it to the uninitialized set
        // the lookup map has only one entry per sequence hash
        if self.lookup_map.contains_key(&sequence_hash) || sequence_hash == 0u64 {
            log::debug!(sequence_hash, "inserted block to uninitialized set");
            self.uninitialized_set.push_back(block);
            return;
        }

        // Insert into timestamp set
        let key = PriorityKey::from(&*block);
        let check_multiple_entries = self.priority_set.insert(key, sequence_hash);
        assert!(
            check_multiple_entries.is_none(),
            "fatal error: multiple entries for the same sequence hash in timestamp set"
        );

        // Add to the lookup map
        let check_multiple_entries = self.lookup_map.insert(sequence_hash, block);
        assert!(
            check_multiple_entries.is_none(),
            "fatal error: multiple entries for the same sequence hash in lookup map"
        );
    }

    fn take_with_sequence_hash(
        &mut self,
        sequence_hash: SequenceHash,
    ) -> Option<PoolValue<KvBlock>> {
        match self.lookup_map.remove(&sequence_hash) {
            Some(block) => {
                // Remove from timestamp set
                self.priority_set.remove(&PriorityKey::from(&*block));
                Some(block)
            }
            None => None,
        }
    }

    fn match_hashes(
        &mut self,
        hashes: Vec<SequenceHash>,
        return_handle: Arc<ReturnHandleImpl>,
    ) -> Vec<PoolItem<KvBlock>> {
        let mut matched_blocks = Vec::with_capacity(hashes.len());

        for hash in hashes {
            if let Some(block) = self.take_with_sequence_hash(hash) {
                matched_blocks.push(self.create_pool_item(block, return_handle.clone()));
            } else {
                break;
            }
        }

        self.available_blocks -= matched_blocks.len() as u64;

        matched_blocks
    }

    fn handle_match_single(&mut self, match_single: MatchSingle) {
        let (hash, return_handle, rx) = match_single.dissolve();

        let matched_blocks = self.match_hashes(vec![hash], return_handle);
        let optional_single = matched_blocks.into_iter().next();

        // Send the result back through the channel
        if rx.send(optional_single).is_err() {
            log::trace!("Failed to send matched block to requester");
        }
    }

    fn handle_match_multiple(&mut self, match_multiple: MatchMultiple) {
        let (hashes, return_handle, rx) = match_multiple.dissolve();

        let matched_blocks = self.match_hashes(hashes, return_handle);

        // Send the matched blocks back through the channel
        if rx.send(matched_blocks).is_err() {
            log::trace!("Failed to send matched blocks to requester");
        }
    }

    fn take(&mut self) -> Option<PoolValue<KvBlock>> {
        // First try uninitialized blocks - these are often part of sequences
        // that have been arranged in the correct order
        if let Some(block) = self.uninitialized_set.pop_front() {
            return Some(block);
        }

        // if we have blocks in the priority set, pop the first (it's sorted by priority)
        // a fatal error will occur if the block is not found in the lookup map
        if let Some((_key, sequence_hash)) = self.priority_set.pop_first() {
            let block = match self.lookup_map.remove(&sequence_hash) {
                Some(block) => block,
                None => {
                    panic!("block from priority set not found in lookup map");
                }
            };

            return Some(block);
        }

        None
    }

    fn handle_take(&mut self, take: Take) {
        let (count, return_handle, tx) = take.dissolve();

        let mut taken_blocks = Vec::with_capacity(count as usize);

        for _ in 0..count {
            if let Some(block) = self.take() {
                taken_blocks.push(self.create_pool_item(block, return_handle.clone()));
            } else {
                break;
            }
        }

        self.available_blocks -= taken_blocks.len() as u64;

        // Send the result back through the channel
        if tx.send(taken_blocks).is_err() {
            log::trace!("Failed to send matched blocks to requester");
        }
    }

    fn handle_match_request(&mut self, match_request: MatchRequest) {
        match match_request {
            MatchRequest::MatchSingle(match_single) => self.handle_match_single(match_single),
            MatchRequest::MatchMultiple(match_multiple) => {
                self.handle_match_multiple(match_multiple)
            }
            MatchRequest::Take(take) => self.handle_take(take),
        }
    }

    fn handle_control_request(&mut self, control_request: ControlRequest) {
        match control_request {
            ControlRequest::Insert(block) => self.handle_insert(block),
            ControlRequest::UpdateSingle(update_single) => self.handle_update_single(update_single),
            ControlRequest::UpdateMultiple(update_multiple) => {
                self.handle_update_multiple(update_multiple)
            }
            ControlRequest::Reset(sequence_hash) => self.handle_reset(sequence_hash),
            ControlRequest::ResetAll => self.handle_reset_all(),
        }
    }

    fn handle_insert(&mut self, block: KvBlock) {
        self.available_blocks += 1;
        self.total_blocks += 1;
        self.insert(PoolValue::Direct(block));
    }

    fn handle_return(&mut self, block: PoolValue<KvBlock>) {
        self.available_blocks += 1;
        self.insert(block);
    }
    fn handle_update_single(&mut self, update: UpdateBlock) {
        self.update_block(vec![update]);
    }

    fn handle_update_multiple(&mut self, updates: Vec<UpdateBlock>) {
        self.update_block(updates);
    }

    fn update_block(&mut self, updates: Vec<UpdateBlock>) {
        for update in updates {
            if let Some(mut block) = self.take_with_sequence_hash(update.hash) {
                if let Some(priority) = update.priority {
                    block.priority = priority;
                }

                // if let Some(deadline) = update.deadline {
                //     block.set_deadline(deadline);
                // }

                self.insert(block);
            }
        }
    }

    fn handle_reset(&mut self, sequence_hashes: Vec<SequenceHash>) {
        for hash in sequence_hashes {
            if let Some(mut block) = self.take_with_sequence_hash(hash) {
                block.reset();
                self.insert(block);
            }
        }
    }

    fn handle_reset_all(&mut self) {
        // for all blocks in the priority set, reset them
        while let Some((_key, sequence_hash)) = self.priority_set.pop_first() {
            if let Some(mut block) = self.lookup_map.remove(&sequence_hash) {
                block.reset();
                self.insert(block);
            } else {
                panic!("block from priority set not found in lookup map");
            }
        }
    }
}

#[async_trait]
impl PoolExt<KvBlock> for AvailableBlocksState {}

#[derive(Dissolve)]
pub struct MatchSingle {
    hash: SequenceHash,
    return_handle: Arc<ReturnHandleImpl>,
    tx: oneshot::Sender<Option<UniqueBlock>>,
}

#[derive(Dissolve)]
pub struct MatchMultiple {
    hashes: Vec<SequenceHash>,
    return_handle: Arc<ReturnHandleImpl>,
    tx: oneshot::Sender<Vec<UniqueBlock>>,
}

#[derive(Dissolve)]
pub struct Take {
    count: u32,
    return_handle: Arc<ReturnHandleImpl>,
    tx: oneshot::Sender<Vec<UniqueBlock>>,
}

pub enum MatchRequest {
    MatchSingle(MatchSingle),
    MatchMultiple(MatchMultiple),
    Take(Take),
}

pub struct UpdateBlock {
    hash: SequenceHash,
    priority: Option<u32>,
    deadline: Option<Instant>,
}

pub enum ControlRequest {
    Insert(KvBlock),
    UpdateSingle(UpdateBlock),
    UpdateMultiple(Vec<UpdateBlock>),
    Reset(Vec<SequenceHash>),
    ResetAll,
}

pub async fn progress_engine(
    match_rx: mpsc::UnboundedReceiver<MatchRequest>,
    return_rx: mpsc::UnboundedReceiver<PoolValue<KvBlock>>,
    ctrl_rx: mpsc::UnboundedReceiver<ControlRequest>,
) {
    let mut match_rx = match_rx;
    let mut return_rx = return_rx;
    let mut ctrl_rx = ctrl_rx;

    let mut state = AvailableBlocksState::default();

    loop {
        tokio::select! {
            biased;

            Some(match_req) = match_rx.recv(), if !match_rx.is_closed() => {
                state.handle_match_request(match_req);
            }

            Some(block) = return_rx.recv(), if !return_rx.is_closed() => {
                state.handle_return(block);
            }

            ctrl_req = ctrl_rx.recv() => {
                match ctrl_req {
                    Some(req) => state.handle_control_request(req),
                    None => {
                        // always flush the control channel before exiting
                        log::trace!("ctrl channel closed, exiting reuse progress loop");
                        break;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_priority_key_ord() {
        let mut map = BTreeMap::new();
        let hash1 = SequenceHash::from(1u64);
        let hash2 = SequenceHash::from(2u64);
        let hash3 = SequenceHash::from(3u64);

        map.insert(
            PriorityKey {
                priority: 0,
                return_tick: 1,
                sequence_hash: hash1,
            },
            "value1",
        );
        map.insert(
            PriorityKey {
                priority: 1,
                return_tick: 0,
                sequence_hash: hash2,
            },
            "value2",
        );
        map.insert(
            PriorityKey {
                priority: 0,
                return_tick: 2,
                sequence_hash: hash3,
            },
            "value3",
        );

        let keys: Vec<_> = map.keys().collect();

        // Priority is the primary sort key (0 before 1)
        assert_eq!(keys[0].priority, 0);
        assert_eq!(keys[1].priority, 0);
        assert_eq!(keys[2].priority, 1);

        // For same priority, return_tick is the secondary sort key
        assert_eq!(keys[0].return_tick, 1);
        assert_eq!(keys[1].return_tick, 2);

        // Test popping from the map to verify ordering
        let (first_key, first_value) = map.pop_first().unwrap();
        assert_eq!(first_key.priority, 0);
        assert_eq!(first_key.return_tick, 1);
        assert_eq!(first_key.sequence_hash, hash1);
        assert_eq!(first_value, "value1");

        let (second_key, second_value) = map.pop_first().unwrap();
        assert_eq!(second_key.priority, 0);
        assert_eq!(second_key.return_tick, 2);
        assert_eq!(second_key.sequence_hash, hash3);
        assert_eq!(second_value, "value3");

        let (third_key, third_value) = map.pop_first().unwrap();
        assert_eq!(third_key.priority, 1);
        assert_eq!(third_key.return_tick, 0);
        assert_eq!(third_key.sequence_hash, hash2);
        assert_eq!(third_value, "value2");

        // Map should now be empty
        assert!(map.is_empty());
    }
}
