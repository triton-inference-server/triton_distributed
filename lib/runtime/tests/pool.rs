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

use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::{Mutex, Notify};
use triton_distributed_runtime::utils::pool::{PoolExt, PoolItem, PoolValue, Returnable};

/// A pool that maintains items in sorted order
pub struct IndexedPool<T: Returnable + Ord + Eq + PartialEq> {
    pool: Arc<Mutex<Vec<PoolValue<T>>>>, // Using Vec instead of VecDeque for sorting
    available: Arc<Notify>,
    capacity: usize,
}

impl<T: Returnable + Ord + Eq + PartialEq> IndexedPool<T> {
    /// Create a new indexed pool with the given initial elements
    pub fn new(mut initial_elements: Vec<PoolValue<T>>) -> Self {
        let capacity = initial_elements.len();

        // Sort the initial elements
        initial_elements.sort_by(|a, b| a.get().cmp(b.get()));

        Self {
            pool: Arc::new(Mutex::new(initial_elements)),
            available: Arc::new(Notify::new()),
            capacity,
        }
    }

    /// Create a new pool with initial boxed elements
    pub fn new_boxed(initial_elements: Vec<Box<T>>) -> Self {
        let initial_values = initial_elements
            .into_iter()
            .map(PoolValue::from_boxed)
            .collect();
        Self::new(initial_values)
    }

    /// Create a new pool with initial direct elements
    pub fn new_direct(initial_elements: Vec<T>) -> Self {
        let initial_values = initial_elements
            .into_iter()
            .map(PoolValue::from_direct)
            .collect();
        Self::new(initial_values)
    }

    /// Get a snapshot of the current pool contents for testing
    pub async fn get_contents(&self) -> Vec<T>
    where
        T: Clone,
    {
        let pool = self.pool.lock().await;
        pool.iter().map(|v| v.get().clone()).collect()
    }
}

#[async_trait]
impl<T: Returnable + Ord + Eq + PartialEq + Send + Sync + 'static> PoolExt<T> for IndexedPool<T> {
    async fn try_acquire(&self) -> Option<PoolItem<T, Self>> {
        let mut pool = self.pool.lock().await;
        if pool.is_empty() {
            return None;
        }

        // Take the first (smallest) element
        let value = pool.remove(0);
        // Use the factory method instead of direct construction
        Some(self.create_pool_item(value))
    }

    async fn acquire(&self) -> PoolItem<T, Self> {
        loop {
            if let Some(guard) = self.try_acquire().await {
                return guard;
            }
            self.available.notified().await;
        }
    }

    async fn return_to_pool(&self, value: PoolValue<T>) {
        let mut pool = self.pool.lock().await;

        // Find the correct position to insert the value to maintain sorted order
        let pos = pool
            .binary_search_by(|probe| probe.get().cmp(value.get()))
            .unwrap_or_else(|e| e);

        pool.insert(pos, value);
    }

    fn notify_return(&self) {
        self.available.notify_one();
    }

    fn capacity(&self) -> usize {
        self.capacity
    }
}

impl<T: Returnable + Ord + Eq + PartialEq> Clone for IndexedPool<T> {
    fn clone(&self) -> Self {
        IndexedPool {
            pool: Arc::clone(&self.pool),
            available: Arc::clone(&self.available),
            capacity: self.capacity,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct NonResettableInt(i32);

impl Returnable for NonResettableInt {
    fn on_return(&mut self) {}
}

impl From<i32> for NonResettableInt {
    fn from(value: i32) -> Self {
        NonResettableInt(value)
    }
}

#[tokio::test]
async fn test_indexed_pool_sorting() {
    // Create an indexed pool with unsorted elements
    let initial_elements = vec![
        PoolValue::Direct(NonResettableInt::from(5)),
        PoolValue::Direct(NonResettableInt::from(3)),
        PoolValue::Direct(NonResettableInt::from(1)),
        PoolValue::Direct(NonResettableInt::from(4)),
        PoolValue::Direct(NonResettableInt::from(2)),
    ];
    let pool = IndexedPool::new(initial_elements);

    // Verify initial sorting
    let contents = pool.get_contents().await;
    assert_eq!(
        contents,
        vec![
            NonResettableInt(1),
            NonResettableInt(2),
            NonResettableInt(3),
            NonResettableInt(4),
            NonResettableInt(5)
        ]
    );

    // Acquire an item (should be the smallest)
    let mut item1 = pool.acquire().await;
    assert_eq!(*item1, NonResettableInt(1));

    // Acquire another item
    let mut item2 = pool.acquire().await;
    assert_eq!(*item2, NonResettableInt(2));

    // Modify item1 to be larger than all remaining items
    *item1 = NonResettableInt(10);

    // Return item1 to the pool - should go at the end
    drop(item1);

    // Check the order after returning
    let contents = pool.get_contents().await;
    assert_eq!(
        contents,
        vec![
            NonResettableInt(3),
            NonResettableInt(4),
            NonResettableInt(5),
            NonResettableInt(10)
        ]
    );

    // Modify item2 to be in the middle
    *item2 = NonResettableInt(4);

    // Return item2 to the pool - should be inserted in the middle
    drop(item2);

    // Check the final order
    let contents = pool.get_contents().await;
    assert_eq!(
        contents,
        vec![
            NonResettableInt(3),
            NonResettableInt(4),
            NonResettableInt(4),
            NonResettableInt(5),
            NonResettableInt(10)
        ]
    );
    // Test returning to a different pool
    let pool2 = IndexedPool::new(vec![PoolValue::Direct(NonResettableInt(42))]);

    // Acquire from first pool
    let mut item = pool.acquire().await;
    assert_eq!(*item, NonResettableInt(3));

    // Modify and return to second pool
    *item = NonResettableInt(8);
    item.return_to_different_pool(&pool2).await;

    // Verify item is in the second pool in sorted order
    let contents = pool2.get_contents().await;
    assert_eq!(contents, vec![NonResettableInt(8), NonResettableInt(42)]);
}
