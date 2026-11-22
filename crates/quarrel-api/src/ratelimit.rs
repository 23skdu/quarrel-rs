//! Bounded, in-memory, per-key fixed-window rate limiter.
//!
//! The mutex guards only a handful of integer operations: the lock is never held across an await
//! point, and the number of tracked keys is capped with FIFO eviction so a client cannot grow the
//! map without bound by rotating API keys.

use std::{
    collections::{HashMap, VecDeque},
    sync::{Arc, Mutex, MutexGuard},
    time::{Duration, Instant},
};

use crate::error::ApiError;

/// Default number of requests allowed per key per window.
pub const DEFAULT_REQUEST_LIMIT: u32 = 120;
/// Default length of one rate-limit window.
pub const DEFAULT_WINDOW: Duration = Duration::from_secs(60);
/// Default number of distinct keys tracked before the oldest is evicted.
pub const DEFAULT_MAX_KEYS: usize = 4_096;

/// A fixed-window counter for each client key.
#[derive(Debug, Clone)]
pub struct RateLimiter {
    limit: u32,
    window: Duration,
    max_keys: usize,
    state: Arc<Mutex<LimiterState>>,
}

#[derive(Debug, Default)]
struct LimiterState {
    entries: HashMap<String, Window>,
    order: VecDeque<String>,
}

#[derive(Debug)]
struct Window {
    started: Instant,
    count: u32,
}

impl Default for RateLimiter {
    fn default() -> Self {
        Self::new(DEFAULT_REQUEST_LIMIT, DEFAULT_WINDOW)
    }
}

impl RateLimiter {
    /// Creates a limiter with a custom quota and window and the default key bound.
    #[must_use]
    pub fn new(limit: u32, window: Duration) -> Self {
        Self::with_max_keys(limit, window, DEFAULT_MAX_KEYS)
    }

    /// Creates a limiter with an explicit bound on the number of tracked keys.
    ///
    /// A zero `limit` rejects every request; a zero `max_keys` means no keys can be stored, so
    /// every request is rejected as well.
    #[must_use]
    pub fn with_max_keys(limit: u32, window: Duration, max_keys: usize) -> Self {
        Self {
            limit,
            window,
            max_keys,
            state: Arc::new(Mutex::new(LimiterState::default())),
        }
    }

    /// Records one request for `key`, returning a typed error when the quota is exhausted.
    ///
    /// # Errors
    ///
    /// Returns [`ApiError::RateLimited`] when the key has used its quota in the current window.
    pub fn check(&self, key: &str) -> Result<(), ApiError> {
        let mut state = self.lock();
        let now = Instant::now();

        if let Some(window) = state.entries.get_mut(key) {
            if now.duration_since(window.started) >= self.window {
                window.started = now;
                window.count = 0;
            }
            if window.count >= self.limit {
                return Err(ApiError::RateLimited);
            }
            window.count += 1;
            return Ok(());
        }

        if self.max_keys == 0 {
            return Err(ApiError::RateLimited);
        }
        if state.entries.len() >= self.max_keys
            && let Some(evicted) = state.order.pop_front()
        {
            state.entries.remove(&evicted);
        }
        state.entries.insert(
            key.to_owned(),
            Window {
                started: now,
                count: 1,
            },
        );
        state.order.push_back(key.to_owned());
        if self.limit == 0 {
            return Err(ApiError::RateLimited);
        }
        Ok(())
    }

    /// Number of keys currently tracked; used by the bound test and by diagnostics.
    #[must_use]
    pub fn tracked_keys(&self) -> usize {
        self.lock().entries.len()
    }

    fn lock(&self) -> MutexGuard<'_, LimiterState> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::RateLimiter;

    #[test]
    fn quota_is_enforced_per_key() {
        let limiter = RateLimiter::new(2, Duration::from_secs(60));
        assert!(limiter.check("a").is_ok());
        assert!(limiter.check("a").is_ok());
        assert!(limiter.check("a").is_err());
        assert!(limiter.check("b").is_ok());
        assert_eq!(limiter.tracked_keys(), 2);
    }

    #[test]
    fn window_resets_the_counter() {
        let limiter = RateLimiter::new(1, Duration::from_millis(5));
        assert!(limiter.check("a").is_ok());
        assert!(limiter.check("a").is_err());
        std::thread::sleep(Duration::from_millis(15));
        assert!(limiter.check("a").is_ok());
    }

    #[test]
    fn key_count_is_bounded_with_fifo_eviction() {
        let limiter = RateLimiter::with_max_keys(3, Duration::from_secs(60), 2);
        assert!(limiter.check("a").is_ok());
        assert!(limiter.check("b").is_ok());
        assert_eq!(limiter.tracked_keys(), 2);
        assert!(limiter.check("c").is_ok());
        assert_eq!(limiter.tracked_keys(), 2);
        // "a" was evicted, so it starts with a fresh quota; "b" was evicted by that check and
        // "c" still has two requests left out of its quota of three.
        assert!(limiter.check("a").is_ok());
        assert!(limiter.check("c").is_ok());
        assert!(limiter.check("c").is_ok());
        assert!(limiter.check("c").is_err());
    }

    #[test]
    fn zero_limits_reject_everything() {
        assert!(
            RateLimiter::new(0, Duration::from_secs(1))
                .check("a")
                .is_err()
        );
        assert!(
            RateLimiter::with_max_keys(1, Duration::from_secs(1), 0)
                .check("a")
                .is_err()
        );
    }

    #[tokio::test]
    async fn concurrent_checks_never_exceed_the_quota() {
        let limiter = std::sync::Arc::new(RateLimiter::new(8, Duration::from_secs(60)));
        let barrier = std::sync::Arc::new(tokio::sync::Barrier::new(32));
        let mut tasks = Vec::new();
        for _ in 0..32 {
            let limiter = std::sync::Arc::clone(&limiter);
            let barrier = std::sync::Arc::clone(&barrier);
            tasks.push(tokio::spawn(async move {
                barrier.wait().await;
                limiter.check("shared").is_ok()
            }));
        }
        let accepted = futures::future::join_all(tasks)
            .await
            .into_iter()
            .filter_map(Result::ok)
            .filter(|ok| *ok)
            .count();
        assert_eq!(accepted, 8);
    }
}
