use std::collections::VecDeque;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;
use tokio::sync::{Mutex, Notify};

pub trait Resettable: Send + Sync + 'static {
    fn reset(&mut self) {}
}

/// Enum to hold either a Box<T> or T directly
pub enum PoolValue<T: Resettable> {
    Boxed(Box<T>),
    Direct(T),
}

impl<T: Resettable> PoolValue<T> {
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

    /// Reset the underlying item
    pub fn reset(&mut self) {
        self.get_mut().reset();
    }
}

pub struct Pool<T: Resettable> {
    pool: Arc<Mutex<VecDeque<PoolValue<T>>>>,
    available: Arc<Notify>,
    capacity: usize,
}

impl<T: Resettable> Pool<T> {
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

    pub async fn try_acquire(&self) -> Option<PoolItem<T>> {
        let mut pool = self.pool.lock().await;
        pool.pop_front().map(|value| PoolItem {
            value: Some(value),
            pool: self.pool.clone(),
            available: self.available.clone(),
        })
    }

    pub async fn acquire(&self) -> PoolItem<T> {
        loop {
            if let Some(guard) = self.try_acquire().await {
                return guard;
            }
            self.available.notified().await;
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

impl<T: Resettable> Clone for Pool<T> {
    fn clone(&self) -> Self {
        Pool {
            pool: Arc::clone(&self.pool),
            available: Arc::clone(&self.available),
            capacity: self.capacity,
        }
    }
}

pub struct PoolItem<T: Resettable> {
    value: Option<PoolValue<T>>,
    pool: Arc<Mutex<VecDeque<PoolValue<T>>>>,
    available: Arc<Notify>,
}

impl<T: Resettable> PoolItem<T> {
    /// Convert this unique PoolItem into a shared reference
    pub fn into_shared(self) -> SharedPoolItem<T> {
        SharedPoolItem {
            inner: Arc::new(self),
        }
    }
}

impl<T: Resettable> Deref for PoolItem<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.value.as_ref().unwrap().get()
    }
}

impl<T: Resettable> DerefMut for PoolItem<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.value.as_mut().unwrap().get_mut()
    }
}

impl<T: Resettable> Drop for PoolItem<T> {
    fn drop(&mut self) {
        if let Some(mut value) = self.value.take() {
            value.reset();
            let mut pool = futures::executor::block_on(self.pool.lock());
            pool.push_back(value);
            self.available.notify_one();
        }
    }
}

/// A shared reference to a pooled item
#[derive(Clone)]
pub struct SharedPoolItem<T: Resettable> {
    inner: Arc<PoolItem<T>>,
}

impl<T: Resettable> SharedPoolItem<T> {
    /// Get a reference to the underlying item
    pub fn get(&self) -> &T {
        self.inner.value.as_ref().unwrap().get()
    }
}

impl<T: Resettable> Deref for SharedPoolItem<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.inner.value.as_ref().unwrap().get()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{timeout, Duration};

    // Implement Resettable for u32 just for testing
    impl Resettable for u32 {
        fn reset(&mut self) {
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

        // The last element in `values` should be the one we returned, and it should be reset to 0
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
        assert_eq!(*item, 0); // Value should be reset
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

        // Should get reset value when acquired again
        let item = pool.acquire().await;
        assert_eq!(*item, 0);
    }
}
