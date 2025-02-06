use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ForwardPassMetrics {
    pub request_active_slots: u64,
    pub request_total_slots: u64,
    pub kv_active_blocks: u64,
    pub kv_total_blocks: u64,
}
