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
use std::collections::VecDeque;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;
use tokio::sync::{Mutex, Notify};

/// Trait for items that can be returned to a pool
pub trait Returnable: Send + Sync + 'static {
    /// Called when an item is returned to the pool
    fn on_return(&mut self) {}
}

/// Enum to hold either a Box<T> or T directly
pub enum PoolValue<T: Returnable> {
    Boxed(Box<T>),
    Direct(T),
}

impl<T: Returnable> PoolValue<T> {
    /// Create a new PoolValue from a boxed item
    pub fn from_boxed(value: Box<T>) -> Self {
        PoolValue::Boxed(value)
    }

    /// Create a new PoolValue from a direct item
    pub fn from_direct(value: T) -> Self {
        PoolValue::Direct(value)
    }

    /// Get a reference to the underlying item
    pub fn get(&self) -> &T {
        match self {
            PoolValue::Boxed(boxed) => boxed.as_ref(),
            PoolValue::Direct(direct) => direct,
        }
    }

    /// Get a mutable reference to the underlying item
    pub fn get_mut(&mut self) -> &mut T {
        match self {
            PoolValue::Boxed(boxed) => boxed.as_mut(),
            PoolValue::Direct(direct) => direct,
        }
    }

    /// Call on_return on the underlying item
    pub fn on_return(&mut self) {
        self.get_mut().on_return();
    }
}

// Private module to restrict access to PoolItem constructor
mod private {
    // This type can only be constructed within this module
    #[derive(Clone, Copy)]
    pub struct PoolItemToken(());

    impl PoolItemToken {
        pub(super) fn new() -> Self {
            PoolItemToken(())
        }
    }
}

/// Core trait defining pool operations
#[async_trait]
pub trait PoolExt<T: Returnable>: Clone + Send + Sync + 'static {
    /// Try to acquire an item from the pool
    async fn try_acquire(&self) -> Option<PoolItem<T, Self>>;

    /// Acquire an item from the pool, waiting if necessary
    async fn acquire(&self) -> PoolItem<T, Self>;

    /// Return a value to the pool
    async fn return_to_pool(&self, value: PoolValue<T>);

    /// Notify that an item has been returned
    fn notify_return(&self);

    /// Get the capacity of the pool
    fn capacity(&self) -> usize;

    /// Create a new PoolItem (only available to implementors)
    fn create_pool_item(&self, value: PoolValue<T>) -> PoolItem<T, Self> {
        PoolItem::new(value, self.clone())
    }
}

/// An item borrowed from a pool
pub struct PoolItem<T: Returnable, P: PoolExt<T>> {
    value: Option<PoolValue<T>>,
    pool: P,
    _token: private::PoolItemToken,
}

impl<T: Returnable, P: PoolExt<T>> PoolItem<T, P> {
    /// Create a new PoolItem (only available within this module)
    fn new(value: PoolValue<T>, pool: P) -> Self {
        Self {
            value: Some(value),
            pool,
            _token: private::PoolItemToken::new(),
        }
    }

    /// Convert this unique PoolItem into a shared reference
    pub fn into_shared(self) -> SharedPoolItem<T, P> {
        SharedPoolItem {
            inner: Arc::new(self),
        }
    }

    /// Return this item to a different pool
    pub async fn return_to_different_pool<P2: PoolExt<T>>(mut self, other_pool: &P2) {
        if let Some(mut value) = self.value.take() {
            value.on_return();
            other_pool.return_to_pool(value).await;
            other_pool.notify_return();
            // Prevent the original drop from happening
            std::mem::forget(self);
        }
    }

    /// Get a reference to the pool this item belongs to
    pub fn pool(&self) -> &P {
        &self.pool
    }

    /// Check if this item still contains a value
    pub fn has_value(&self) -> bool {
        self.value.is_some()
    }
}

impl<T: Returnable, P: PoolExt<T>> Deref for PoolItem<T, P> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.value.as_ref().unwrap().get()
    }
}

impl<T: Returnable, P: PoolExt<T>> DerefMut for PoolItem<T, P> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.value.as_mut().unwrap().get_mut()
    }
}

impl<T: Returnable, P: PoolExt<T>> Drop for PoolItem<T, P> {
    fn drop(&mut self) {
        if let Some(mut value) = self.value.take() {
            value.on_return();
            // Use blocking version for drop
            futures::executor::block_on(self.pool.return_to_pool(value));
            self.pool.notify_return();
        }
    }
}

/// A shared reference to a pooled item
#[derive(Clone)]
pub struct SharedPoolItem<T: Returnable, P: PoolExt<T>> {
    inner: Arc<PoolItem<T, P>>,
}

impl<T: Returnable, P: PoolExt<T>> SharedPoolItem<T, P> {
    /// Get a reference to the underlying item
    pub fn get(&self) -> &T {
        self.inner.value.as_ref().unwrap().get()
    }

    /// Get a reference to the pool this item belongs to
    pub fn pool(&self) -> &P {
        &self.inner.pool
    }

    /// Return this shared item to a different pool
    pub async fn return_to_different_pool<P2: PoolExt<T>>(self, other_pool: &P2) -> bool {
        // Only return if we're the last reference
        if Arc::strong_count(&self.inner) == 1 {
            if let Some(mut value) = Arc::get_mut(&mut self.inner.clone()).unwrap().value.take() {
                value.on_return();
                other_pool.return_to_pool(value).await;
                other_pool.notify_return();
                return true;
            }
        }
        false
    }
}

impl<T: Returnable, P: PoolExt<T>> Deref for SharedPoolItem<T, P> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.inner.value.as_ref().unwrap().get()
    }
}

/// Standard pool implementation
pub struct Pool<T: Returnable> {
    pool: Arc<Mutex<VecDeque<PoolValue<T>>>>,
    available: Arc<Notify>,
    capacity: usize,
}

impl<T: Returnable> Pool<T> {
    /// Create a new pool with the given initial elements
    pub fn new(initial_elements: Vec<PoolValue<T>>) -> Self {
        let capacity = initial_elements.len();
        let pool = initial_elements
            .into_iter()
            .collect::<VecDeque<PoolValue<T>>>();
        Self {
            pool: Arc::new(Mutex::new(pool)),
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
}

#[async_trait]
impl<T: Returnable> PoolExt<T> for Pool<T> {
    async fn try_acquire(&self) -> Option<PoolItem<T, Self>> {
        let mut pool = self.pool.lock().await;
        pool.pop_front().map(|value| self.create_pool_item(value))
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
        pool.push_back(value);
    }

    fn notify_return(&self) {
        self.available.notify_one();
    }

    fn capacity(&self) -> usize {
        self.capacity
    }
}

impl<T: Returnable> Clone for Pool<T> {
    fn clone(&self) -> Self {
        Pool {
            pool: Arc::clone(&self.pool),
            available: Arc::clone(&self.available),
            capacity: self.capacity,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{timeout, Duration};

    // Implement Returnable for u32 just for testing
    impl Returnable for u32 {
        fn on_return(&mut self) {
            *self = 0;
            println!("Resetting u32 to 0");
        }
    }

    #[tokio::test]
    async fn test_acquire_release() {
        let initial_elements = vec![
            PoolValue::Direct(1),
            PoolValue::Direct(2),
            PoolValue::Direct(3),
            PoolValue::Direct(4),
            PoolValue::Direct(5),
        ];
        let pool = Pool::new(initial_elements);

        // Acquire an element from the pool
        if let Some(mut item) = pool.try_acquire().await {
            assert_eq!(*item, 1); // It should be the first element we put in

            // Modify the value
            *item += 10;
            assert_eq!(*item, 11);

            // The item will be dropped at the end of this scope,
            // and the value will be returned to the pool
        }

        // Acquire all remaining elements and the one we returned
        let mut values = Vec::new();
        let mut items = Vec::new();
        while let Some(item) = pool.try_acquire().await {
            values.push(*item);
            items.push(item);
        }

        // The last element in `values` should be the one we returned, and it should be on_return to 0
        assert_eq!(values, vec![2, 3, 4, 5, 0]);

        // Test the awaitable acquire
        let pool_clone = pool.clone();
        let task = tokio::spawn(async move {
            let first_acquired = pool_clone.acquire().await;
            assert_eq!(*first_acquired, 0);
        });

        timeout(Duration::from_secs(1), task)
            .await
            .expect_err("Expected timeout");

        // Drop the guards to return the PoolItems to the pool.
        items.clear();

        let pool_clone = pool.clone();
        let task = tokio::spawn(async move {
            let first_acquired = pool_clone.acquire().await;
            assert_eq!(*first_acquired, 0);
        });

        // Now the task should be able to finish.
        timeout(Duration::from_secs(1), task)
            .await
            .expect("Task did not complete in time")
            .unwrap();
    }

    #[tokio::test]
    async fn test_shared_items() {
        let initial_elements = vec![
            PoolValue::Direct(1),
            // PoolValue::Direct(2),
            // PoolValue::Direct(3),
        ];
        let pool = Pool::new(initial_elements);

        // Acquire and convert to shared
        let mut item = pool.acquire().await;
        *item += 10; // Modify before sharing
        let shared = item.into_shared();
        assert_eq!(*shared, 11);

        // Create a clone of the shared item
        let shared_clone = shared.clone();
        assert_eq!(*shared_clone, 11);

        // Drop the original shared item
        drop(shared);

        // Clone should still be valid
        assert_eq!(*shared_clone, 11);

        // Drop the clone
        drop(shared_clone);

        // Now we should be able to acquire the item again
        let item = pool.acquire().await;
        assert_eq!(*item, 0); // Value should be on_return
    }

    #[tokio::test]
    async fn test_boxed_values() {
        let initial_elements = vec![
            PoolValue::Boxed(Box::new(1)),
            // PoolValue::Boxed(Box::new(2)),
            // PoolValue::Boxed(Box::new(3)),
        ];
        let pool = Pool::new(initial_elements);

        // Acquire an element from the pool
        let mut item = pool.acquire().await;
        assert_eq!(*item, 1);

        // Modify and return to pool
        *item += 10;
        drop(item);

        // Should get on_return value when acquired again
        let item = pool.acquire().await;
        assert_eq!(*item, 0);
    }

    #[tokio::test]
    async fn test_return_to_different_pool() {
        let pool1 = Pool::new(vec![PoolValue::Direct(1)]);
        let pool2 = pool1.clone();

        let item = pool1.acquire().await;
        assert_eq!(*item, 1);

        item.return_to_different_pool(&pool2).await;

        let item2 = pool2.acquire().await;
        assert_eq!(*item2, 0);
    }

    #[tokio::test]
    async fn test_pool_item_creation() {
        let pool = Pool::new(vec![PoolValue::Direct(1)]);

        // This works - acquiring from the pool
        let item = pool.acquire().await;
        assert_eq!(*item, 1);

        // This would not compile - can't create PoolItem directly
        // let invalid_item = PoolItem {
        //     value: Some(PoolValue::Direct(2)),
        //     pool: pool.clone(),
        //     _token: /* can't create this */
        // };
    }
}
