//! Bounded reuse pool for generate-response builders.

use std::{
    collections::HashSet,
    sync::{
        Arc, Mutex, MutexGuard, PoisonError,
        atomic::{AtomicU64, Ordering},
    },
};

use arrow::record_batch::RecordBatch;
use quarrel_core::{RequestId, Token};

use crate::{FlightError, GenerateResponseBuilder};

static NEXT_POOL_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_BUILDER_ID: AtomicU64 = AtomicU64::new(1);

/// A bounded pool of reusable [`GenerateResponseBuilder`] instances.
///
/// Arrow output buffers are immutable and necessarily per-response: a
/// [`RecordBatch`] column owns its buffers and cannot be overwritten while a
/// consumer may still hold the batch. This pool therefore does **not** recycle
/// output buffers. What it recycles is the builder's small typed scratch
/// vectors (`request_id`, sequences, token ids, logprobs, finish flags) and the
/// prebuilt constant columns whose `Arc` handles are shared by every response.
/// After warm-up, acquiring a builder from the pool does not reallocate that
/// scratch capacity.
///
/// The pool is bounded by the maximum number of concurrent loans. When all
/// builders are checked out, [`ResponseBatchPool::acquire`] returns
/// [`FlightError::PoolExhausted`] rather than growing without limit. Return a
/// builder with [`PooledResponseBuilder::release`] or
/// [`ResponseBatchPool::release`]; dropping a [`PooledResponseBuilder`] also
/// returns it automatically, so an error while encoding cannot leak capacity.
///
/// The pool is safe to share across threads.
#[derive(Debug)]
pub struct ResponseBatchPool {
    inner: Arc<PoolInner>,
}

#[derive(Debug)]
struct PoolInner {
    id: u64,
    capacity: usize,
    state: Mutex<PoolState>,
}

#[derive(Debug)]
struct PoolState {
    idle: Vec<GenerateResponseBuilder>,
    loaned: HashSet<u64>,
}

/// A checked-out [`GenerateResponseBuilder`] owned by a [`ResponseBatchPool`].
///
/// Borrow the contained builder with [`PooledResponseBuilder::builder`] or
/// [`PooledResponseBuilder::builder_mut`], or call the forwarded
/// [`PooledResponseBuilder::encode`]. Dropping the guard returns the builder to
/// its pool; call [`PooledResponseBuilder::release`] to return it explicitly.
#[derive(Debug)]
pub struct PooledResponseBuilder {
    pool: Arc<PoolInner>,
    builder: Option<GenerateResponseBuilder>,
}

impl ResponseBatchPool {
    /// Create a pool that lends at most `capacity` builders at once.
    ///
    /// # Errors
    ///
    /// Returns [`FlightError::InvalidArgument`] if `capacity` is zero.
    pub fn new(capacity: usize) -> Result<Self, FlightError> {
        if capacity == 0 {
            return Err(FlightError::InvalidArgument(
                "response builder pool capacity must be greater than zero".into(),
            ));
        }
        Ok(Self {
            inner: Arc::new(PoolInner {
                id: NEXT_POOL_ID.fetch_add(1, Ordering::Relaxed),
                capacity,
                state: Mutex::new(PoolState {
                    idle: Vec::new(),
                    loaned: HashSet::new(),
                }),
            }),
        })
    }

    /// Maximum number of builders this pool may lend concurrently.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.inner.capacity
    }

    /// Number of builders currently idle and available for reuse.
    #[must_use]
    pub fn available(&self) -> usize {
        self.lock().idle.len()
    }

    /// Number of builders currently checked out.
    #[must_use]
    pub fn in_use(&self) -> usize {
        self.lock().loaned.len()
    }

    /// Check out a builder, reusing an idle one or creating a new one below capacity.
    ///
    /// # Errors
    ///
    /// Returns [`FlightError::PoolExhausted`] when every builder is already
    /// loaned out, or an underlying [`FlightError`] if a new builder cannot be
    /// initialised.
    pub fn acquire(&self) -> Result<PooledResponseBuilder, FlightError> {
        let mut state = self.lock();
        if let Some(builder) = state.idle.pop() {
            let builder_id = builder.builder_id();
            if !state.loaned.insert(builder_id) {
                return Err(FlightError::InvalidBuilderRelease(
                    "idle builder was already marked as loaned".into(),
                ));
            }
            return Ok(PooledResponseBuilder {
                pool: Arc::clone(&self.inner),
                builder: Some(builder),
            });
        }
        if state.loaned.len() >= self.inner.capacity {
            return Err(FlightError::PoolExhausted {
                capacity: self.inner.capacity,
            });
        }
        let builder_id = NEXT_BUILDER_ID.fetch_add(1, Ordering::Relaxed);
        let builder = GenerateResponseBuilder::new()?.with_identity(self.inner.id, builder_id);
        if !state.loaned.insert(builder_id) {
            return Err(FlightError::InvalidBuilderRelease(
                "pool builder id collision".into(),
            ));
        }
        Ok(PooledResponseBuilder {
            pool: Arc::clone(&self.inner),
            builder: Some(builder),
        })
    }

    /// Return a checked-out builder to this pool.
    ///
    /// Passing a guard issued by another pool returns it to its actual owner
    /// and reports [`FlightError::InvalidBuilderRelease`].
    ///
    /// # Errors
    ///
    /// Returns [`FlightError::InvalidBuilderRelease`] if the guard was not
    /// issued by this pool or is in an invalid state.
    pub fn release(&self, guard: PooledResponseBuilder) -> Result<(), FlightError> {
        let owner = Arc::clone(&guard.pool);
        if !Arc::ptr_eq(&owner, &self.inner) {
            // Do not strand capacity in the pool that actually issued it.
            owner.reclaim(guard)?;
            return Err(FlightError::InvalidBuilderRelease(
                "builder belongs to a different response pool".into(),
            ));
        }
        owner.reclaim(guard)
    }

    fn lock(&self) -> MutexGuard<'_, PoolState> {
        self.inner
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }
}

impl PooledResponseBuilder {
    /// Borrow the pooled builder.
    ///
    /// # Errors
    ///
    /// Returns [`FlightError::InvalidBuilderRelease`] if the builder has
    /// already been returned to its pool.
    pub fn builder(&self) -> Result<&GenerateResponseBuilder, FlightError> {
        self.builder.as_ref().ok_or_else(|| {
            FlightError::InvalidBuilderRelease("builder was already returned to its pool".into())
        })
    }

    /// Mutably borrow the pooled builder.
    ///
    /// # Errors
    ///
    /// Returns [`FlightError::InvalidBuilderRelease`] if the builder has
    /// already been returned to its pool.
    pub fn builder_mut(&mut self) -> Result<&mut GenerateResponseBuilder, FlightError> {
        self.builder.as_mut().ok_or_else(|| {
            FlightError::InvalidBuilderRelease("builder was already returned to its pool".into())
        })
    }

    /// Encode one response token using the pooled builder.
    ///
    /// # Errors
    ///
    /// Propagates builder-state and Arrow encoding errors.
    pub fn encode(
        &mut self,
        request_id: &RequestId,
        sequence: u64,
        token: &Token,
        finished: bool,
    ) -> Result<RecordBatch, FlightError> {
        self.builder_mut()?
            .encode(request_id, sequence, token, finished)
    }

    /// Scratch capacity of the underlying builder.
    ///
    /// # Errors
    ///
    /// Returns [`FlightError::InvalidBuilderRelease`] if the builder has
    /// already been returned to its pool.
    pub fn scratch_capacity(&self) -> Result<usize, FlightError> {
        Ok(self.builder()?.scratch_capacity())
    }

    /// Return this builder to the pool that issued it.
    ///
    /// # Errors
    ///
    /// Returns [`FlightError::InvalidBuilderRelease`] if the pooled builder is
    /// in an invalid state.
    pub fn release(self) -> Result<(), FlightError> {
        let pool = Arc::clone(&self.pool);
        pool.reclaim(self)
    }
}

impl PoolInner {
    fn reclaim(&self, mut guard: PooledResponseBuilder) -> Result<(), FlightError> {
        let builder = guard.builder.take().ok_or_else(|| {
            FlightError::InvalidBuilderRelease("builder was already returned to its pool".into())
        })?;
        self.reclaim_builder(builder)
    }

    fn reclaim_builder(&self, builder: GenerateResponseBuilder) -> Result<(), FlightError> {
        match builder.pool_id() {
            None => {
                return Err(FlightError::InvalidBuilderRelease(
                    "builder was not issued by a response pool".into(),
                ));
            }
            Some(id) if id != self.id => {
                return Err(FlightError::InvalidBuilderRelease(
                    "builder was issued by a different response pool".into(),
                ));
            }
            Some(_) => {}
        }
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        if !state.loaned.remove(&builder.builder_id()) {
            return Err(FlightError::InvalidBuilderRelease(
                "builder was not checked out from this pool".into(),
            ));
        }
        state.idle.push(builder);
        Ok(())
    }
}

impl Drop for PooledResponseBuilder {
    fn drop(&mut self) {
        if let Some(builder) = self.builder.take() {
            // Dropping a guard must never panic; invalid state simply drops the
            // builder instead of returning it to the pool.
            let _ = self.pool.reclaim_builder(builder);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reclaim_rejects_foreign_and_unloaned_builders() {
        let pool = ResponseBatchPool::new(2).expect("valid pool");
        let foreign = GenerateResponseBuilder::new().expect("standalone builder");
        let err = pool
            .inner
            .reclaim_builder(foreign)
            .expect_err("foreign builder must be rejected");
        assert!(matches!(err, FlightError::InvalidBuilderRelease(_)));

        let guard = pool.acquire().expect("first builder");
        let borrowed = guard.builder().expect("live guard");
        let builder_id = borrowed.builder_id();
        let unloaned = GenerateResponseBuilder::new()
            .expect("standalone builder")
            .with_identity(pool.inner.id, builder_id.wrapping_add(1_000));
        let err = pool
            .inner
            .reclaim_builder(unloaned)
            .expect_err("unloaned builder must be rejected");
        assert!(matches!(err, FlightError::InvalidBuilderRelease(_)));
        assert_eq!(pool.in_use(), 1);

        guard.release().expect("valid release");
        assert_eq!(pool.in_use(), 0);
        assert_eq!(pool.available(), 1);
    }
}
