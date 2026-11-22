//! Bounded paged KV cache and an LRU radix prefix cache.
//!
//! [`PagedKvCache`] owns a fixed number of fixed-size blocks. A sequence is an
//! opaque `u64` handle that owns zero or more blocks; blocks are returned to a
//! free list by [`PagedKvCache::rollback`] and [`PagedKvCache::free`]. The
//! caller keeps the association between a `RequestId` and a sequence id.
//!
//! [`RadixPrefixCache`] maps token sequences to `u64` payloads and answers
//! longest-prefix queries. Both structures are synchronous and use only safe
//! standard-library collections.

use crate::EngineError;
use quarrel_core::RequestId;
use std::collections::HashMap;

/// A bounded cache of fixed-size token blocks.
#[derive(Debug)]
pub struct PagedKvCache {
    block_size: usize,
    capacity: usize,
    blocks: Vec<Option<Vec<i32>>>,
    free_blocks: Vec<usize>,
    sequences: HashMap<u64, Sequence>,
    next_sequence: u64,
}

#[derive(Debug)]
struct Sequence {
    blocks: Vec<usize>,
    length: usize,
}

impl PagedKvCache {
    /// Creates an empty cache with `capacity` blocks of `block_size` tokens.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::InvalidBlockSize`] if `block_size` is zero, and
    /// [`EngineError::CacheInvariant`] if `capacity` is zero.
    pub fn new(block_size: usize, capacity: usize) -> Result<Self, EngineError> {
        if block_size == 0 {
            return Err(EngineError::InvalidBlockSize);
        }
        if capacity == 0 {
            return Err(EngineError::CacheInvariant {
                reason: "capacity must be greater than zero",
            });
        }
        Ok(Self {
            block_size,
            capacity,
            blocks: vec![None; capacity],
            free_blocks: (0..capacity).rev().collect(),
            sequences: HashMap::new(),
            next_sequence: 1,
        })
    }

    /// Reserves room for `token_count` tokens and returns a new sequence id.
    ///
    /// The sequence starts empty; use [`PagedKvCache::append`] to fill it. The
    /// `request_id` identifies the owning request to the caller but is not
    /// stored by the cache.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::CapacityExceeded`] when the cache is too small or
    /// too fragmented for the requested token count.
    pub fn allocate(
        &mut self,
        _request_id: &RequestId,
        token_count: usize,
    ) -> Result<u64, EngineError> {
        let required = token_count.div_ceil(self.block_size);
        if required > self.capacity {
            return Err(self.capacity_error(token_count));
        }
        if self.free_blocks.len() < required {
            return Err(EngineError::CacheInvariant {
                reason: "not enough free blocks for allocation",
            });
        }

        let mut owned = Vec::with_capacity(required);
        for _ in 0..required {
            let index = self.free_blocks.pop().ok_or(EngineError::CacheInvariant {
                reason: "free list disagrees with block count",
            })?;
            self.blocks[index] = Some(vec![0; self.block_size]);
            owned.push(index);
        }

        let seq = self.next_sequence;
        self.next_sequence =
            self.next_sequence
                .checked_add(1)
                .ok_or(EngineError::CacheInvariant {
                    reason: "sequence id space exhausted",
                })?;
        self.sequences.insert(
            seq,
            Sequence {
                blocks: owned,
                length: 0,
            },
        );
        Ok(seq)
    }

    /// Appends `tokens` to the logical token stream of `seq`.
    ///
    /// Extra blocks are taken from the free list as needed. Appending an empty
    /// slice is a no-op.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::UnknownSequence`] for an unknown id and
    /// [`EngineError::CapacityExceeded`] if no free block is available.
    pub fn append(&mut self, seq: u64, tokens: &[i32]) -> Result<(), EngineError> {
        if tokens.is_empty() {
            if !self.sequences.contains_key(&seq) {
                return Err(EngineError::UnknownSequence { seq });
            }
            return Ok(());
        }

        let (length, owned_blocks) = {
            let sequence = self
                .sequences
                .get(&seq)
                .ok_or(EngineError::UnknownSequence { seq })?;
            (sequence.length, sequence.blocks.len())
        };
        let final_length = length
            .checked_add(tokens.len())
            .ok_or(EngineError::CacheInvariant {
                reason: "sequence length overflow",
            })?;
        let required_blocks = final_length.div_ceil(self.block_size);
        let extra_blocks = required_blocks.saturating_sub(owned_blocks);
        if extra_blocks > self.free_blocks.len() {
            return Err(self.capacity_error(final_length));
        }

        let mut new_blocks = Vec::with_capacity(extra_blocks);
        for _ in 0..extra_blocks {
            let index = self.free_blocks.pop().ok_or(EngineError::CacheInvariant {
                reason: "free list disagrees with block count",
            })?;
            self.blocks[index] = Some(vec![0; self.block_size]);
            new_blocks.push(index);
        }

        let sequence = self
            .sequences
            .get_mut(&seq)
            .ok_or(EngineError::UnknownSequence { seq })?;
        sequence.blocks.extend(new_blocks);
        for &token in tokens {
            let position = sequence.length;
            let block_slot = position / self.block_size;
            let offset = position % self.block_size;
            let block_index = sequence.blocks[block_slot];
            let block = self.blocks[block_index]
                .as_mut()
                .ok_or(EngineError::CacheInvariant {
                    reason: "owned block is not allocated",
                })?;
            *block.get_mut(offset).ok_or(EngineError::CacheInvariant {
                reason: "token offset outside block",
            })? = token;
            sequence.length += 1;
        }
        Ok(())
    }

    /// Removes the last `token_count` tokens and releases any unused blocks.
    ///
    /// Rolling back more tokens than the sequence holds is an error and leaves
    /// the sequence unchanged. Rolling back to zero keeps the sequence alive
    /// with no owned blocks.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::UnknownSequence`] for an unknown id and
    /// [`EngineError::InvalidRollback`] for an oversized rollback.
    pub fn rollback(&mut self, seq: u64, token_count: usize) -> Result<(), EngineError> {
        let (available, keep_blocks) = {
            let sequence = self
                .sequences
                .get(&seq)
                .ok_or(EngineError::UnknownSequence { seq })?;
            if token_count > sequence.length {
                return Err(EngineError::InvalidRollback {
                    requested: token_count,
                    available: sequence.length,
                });
            }
            let remaining = sequence.length - token_count;
            (sequence.length, remaining.div_ceil(self.block_size))
        };

        let sequence = self
            .sequences
            .get_mut(&seq)
            .ok_or(EngineError::UnknownSequence { seq })?;
        sequence.length -= token_count;
        while sequence.blocks.len() > keep_blocks {
            if let Some(index) = sequence.blocks.pop() {
                self.blocks[index] = None;
                self.free_blocks.push(index);
            }
        }
        debug_assert!(available >= token_count);
        Ok(())
    }

    /// Frees `seq` and returns all of its blocks to the cache.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::UnknownSequence`] if the id is unknown or has
    /// already been freed.
    pub fn free(&mut self, seq: u64) -> Result<(), EngineError> {
        let sequence = self
            .sequences
            .remove(&seq)
            .ok_or(EngineError::UnknownSequence { seq })?;
        for index in sequence.blocks {
            self.blocks[index] = None;
            self.free_blocks.push(index);
        }
        Ok(())
    }

    /// Returns the number of blocks currently owned by sequences.
    #[must_use]
    pub fn blocks_used(&self) -> usize {
        self.capacity - self.free_blocks.len()
    }

    /// Returns the total number of blocks in the cache.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the number of tokens stored in each block.
    #[must_use]
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Copies the logical token stream of `seq` into a new vector.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::UnknownSequence`] for an unknown id and
    /// [`EngineError::CacheInvariant`] if an owned block is missing.
    pub fn snapshot(&self, seq: u64) -> Result<Vec<i32>, EngineError> {
        let sequence = self
            .sequences
            .get(&seq)
            .ok_or(EngineError::UnknownSequence { seq })?;
        let mut tokens = Vec::with_capacity(sequence.length);
        for &block_index in &sequence.blocks {
            let remaining = sequence.length - tokens.len();
            if remaining == 0 {
                break;
            }
            let block = self.blocks[block_index]
                .as_ref()
                .ok_or(EngineError::CacheInvariant {
                    reason: "owned block is not allocated",
                })?;
            let take = remaining.min(self.block_size);
            let slice = block.get(..take).ok_or(EngineError::CacheInvariant {
                reason: "owned block is shorter than block_size",
            })?;
            tokens.extend_from_slice(slice);
        }
        Ok(tokens)
    }

    fn capacity_error(&self, requested: usize) -> EngineError {
        EngineError::CapacityExceeded {
            capacity: self.capacity.saturating_mul(self.block_size),
            requested,
        }
    }
}

#[derive(Debug)]
struct PrefixEntry {
    tokens: Vec<i32>,
    value: u64,
    last_used: u64,
}

/// An LRU cache mapping token sequences to `u64` payloads.
///
/// Inserting an identical sequence replaces its payload and refreshes its
/// recency. When the cache is full, the least recently used entry is evicted;
/// either way `insert` returns the displaced payload, if any. Prefix lookups
/// also refresh recency, so a popular prefix is less likely to be evicted.
#[derive(Debug)]
pub struct RadixPrefixCache {
    capacity: usize,
    entries: Vec<PrefixEntry>,
    clock: u64,
}

impl RadixPrefixCache {
    /// Creates a cache that holds at most `capacity` sequences.
    ///
    /// A capacity of zero is allowed; such a cache simply never stores an
    /// entry.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            entries: Vec::new(),
            clock: 0,
        }
    }

    /// Inserts `tokens` with `value`, returning any displaced payload.
    ///
    /// The returned value is the previous payload for an identical key, or the
    /// evicted payload when a new key displaces the least recently used entry.
    pub fn insert(&mut self, tokens: &[i32], value: u64) -> Option<u64> {
        if self.capacity == 0 {
            return None;
        }

        if let Some(position) = self
            .entries
            .iter()
            .position(|entry| entry.tokens.as_slice() == tokens)
        {
            let displaced = self.entries[position].value;
            self.entries[position].value = value;
            self.touch(position);
            return Some(displaced);
        }

        let mut displaced = None;
        if self.entries.len() == self.capacity {
            let victim = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(position, _)| position);
            if let Some(position) = victim {
                displaced = Some(self.entries[position].value);
                self.entries.remove(position);
            }
        }

        self.clock = self.clock.wrapping_add(1);
        self.entries.push(PrefixEntry {
            tokens: tokens.to_vec(),
            value,
            last_used: self.clock,
        });
        displaced
    }

    /// Returns the longest stored prefix of `tokens` and its payload.
    ///
    /// A matching empty prefix counts when an empty key was inserted. Ties are
    /// broken in favour of the most recently used entry, which is then touched.
    pub fn longest_prefix(&mut self, tokens: &[i32]) -> Option<(usize, u64)> {
        let mut best_position: Option<usize> = None;
        let mut best_length = 0usize;
        let mut best_recency = 0u64;
        for (position, entry) in self.entries.iter().enumerate() {
            if entry.tokens.is_empty() && !tokens.is_empty() {
                continue;
            }
            let matched = entry
                .tokens
                .iter()
                .zip(tokens)
                .take_while(|(left, right)| left == right)
                .count();
            let eligible = matched > 0 || (entry.tokens.is_empty() && tokens.is_empty());
            if !eligible {
                continue;
            }
            let better = best_position.is_none()
                || matched > best_length
                || (matched == best_length && entry.last_used > best_recency);
            if better {
                best_position = Some(position);
                best_length = matched;
                best_recency = entry.last_used;
            }
        }

        let position = best_position?;
        self.touch(position);
        Some((best_length, self.entries[position].value))
    }

    /// Returns the number of entries currently stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` when no entries are stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Removes all entries without changing the capacity.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    fn touch(&mut self, position: usize) {
        self.clock = self.clock.wrapping_add(1);
        if let Some(entry) = self.entries.get_mut(position) {
            entry.last_used = self.clock;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{PagedKvCache, RadixPrefixCache};
    use crate::EngineError;
    use proptest::prelude::*;
    use quarrel_core::RequestId;

    fn request() -> RequestId {
        RequestId::new("kv-test").expect("non-empty request id")
    }

    #[test]
    fn new_validates_geometry() {
        assert!(matches!(
            PagedKvCache::new(0, 4),
            Err(EngineError::InvalidBlockSize)
        ));
        assert!(PagedKvCache::new(4, 0).is_err());
        let cache = PagedKvCache::new(4, 2).expect("valid cache");
        assert_eq!(cache.block_size(), 4);
        assert_eq!(cache.capacity(), 2);
        assert_eq!(cache.blocks_used(), 0);
    }

    #[test]
    fn allocate_append_and_snapshot() {
        let mut cache = PagedKvCache::new(4, 3).expect("valid cache");
        let seq = cache.allocate(&request(), 5).expect("allocation");
        assert_eq!(cache.blocks_used(), 2);
        assert_eq!(cache.snapshot(seq), Ok(Vec::new()));

        cache.append(seq, &[1, 2, 3, 4, 5]).expect("append");
        assert_eq!(cache.snapshot(seq), Ok(vec![1, 2, 3, 4, 5]));
        assert_eq!(cache.blocks_used(), 2);

        cache
            .append(seq, &[6, 7, 8, 9])
            .expect("append across block");
        assert_eq!(cache.snapshot(seq), Ok(vec![1, 2, 3, 4, 5, 6, 7, 8, 9]));
        assert_eq!(cache.blocks_used(), 3);
    }

    #[test]
    fn append_rejects_overflow_without_changing_state() {
        let mut cache = PagedKvCache::new(2, 2).expect("valid cache");
        let seq = cache.allocate(&request(), 4).expect("allocation");
        cache.append(seq, &[1, 2, 3, 4]).expect("append");
        assert!(matches!(
            cache.append(seq, &[5]),
            Err(EngineError::CapacityExceeded { .. })
        ));
        assert_eq!(cache.snapshot(seq), Ok(vec![1, 2, 3, 4]));
    }

    #[test]
    fn rollback_frees_blocks_and_validates_amount() {
        let mut cache = PagedKvCache::new(4, 3).expect("valid cache");
        let seq = cache.allocate(&request(), 9).expect("allocation");
        cache
            .append(seq, &[1, 2, 3, 4, 5, 6, 7, 8, 9])
            .expect("append");
        assert_eq!(cache.blocks_used(), 3);

        assert!(matches!(
            cache.rollback(seq, 10),
            Err(EngineError::InvalidRollback {
                requested: 10,
                available: 9
            })
        ));
        cache.rollback(seq, 6).expect("rollback");
        assert_eq!(cache.snapshot(seq), Ok(vec![1, 2, 3]));
        assert_eq!(cache.blocks_used(), 1);

        cache.rollback(seq, 3).expect("rollback to empty");
        assert_eq!(cache.blocks_used(), 0);
        assert_eq!(cache.snapshot(seq), Ok(Vec::new()));
    }

    #[test]
    fn free_returns_blocks_and_rejects_double_free() {
        let mut cache = PagedKvCache::new(2, 2).expect("valid cache");
        let first = cache.allocate(&request(), 2).expect("allocation");
        let second = cache.allocate(&request(), 2).expect("allocation");
        assert_eq!(cache.blocks_used(), 2);
        cache.free(first).expect("free");
        assert_eq!(cache.blocks_used(), 1);
        assert!(matches!(
            cache.free(first),
            Err(EngineError::UnknownSequence { .. })
        ));
        cache.free(second).expect("free");
        assert_eq!(cache.blocks_used(), 0);
    }

    #[test]
    fn sequences_are_independent() {
        let mut cache = PagedKvCache::new(2, 4).expect("valid cache");
        let first = cache.allocate(&request(), 2).expect("allocation");
        let second = cache.allocate(&request(), 2).expect("allocation");
        assert_ne!(first, second);
        cache.append(first, &[10, 11]).expect("append");
        cache.append(second, &[20, 21]).expect("append");
        assert_eq!(cache.snapshot(first), Ok(vec![10, 11]));
        assert_eq!(cache.snapshot(second), Ok(vec![20, 21]));
        cache.rollback(first, 1).expect("rollback");
        assert_eq!(cache.snapshot(second), Ok(vec![20, 21]));
    }

    #[test]
    fn unknown_sequence_is_reported_by_every_operation() {
        let mut cache = PagedKvCache::new(2, 2).expect("valid cache");
        assert!(matches!(
            cache.append(99, &[1]),
            Err(EngineError::UnknownSequence { .. })
        ));
        assert!(matches!(
            cache.append(99, &[]),
            Err(EngineError::UnknownSequence { .. })
        ));
        assert!(matches!(
            cache.rollback(99, 0),
            Err(EngineError::UnknownSequence { .. })
        ));
        assert!(matches!(
            cache.free(99),
            Err(EngineError::UnknownSequence { .. })
        ));
        assert!(matches!(
            cache.snapshot(99),
            Err(EngineError::UnknownSequence { .. })
        ));
    }

    #[test]
    fn radix_prefix_basics() {
        let mut cache = RadixPrefixCache::new(4);
        assert!(cache.is_empty());
        assert_eq!(cache.insert(&[1, 2, 3], 10), None);
        assert_eq!(cache.insert(&[1, 2, 4], 11), None);
        assert_eq!(cache.longest_prefix(&[1, 2, 3, 9]), Some((3, 10)));
        // The previous lookup refreshed entry `[1, 2, 3]`, so it wins the tie.
        assert_eq!(cache.longest_prefix(&[1, 2]), Some((2, 10)));
        assert_eq!(cache.longest_prefix(&[9]), None);
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn radix_prefix_replacement_and_lru_eviction() {
        let mut cache = RadixPrefixCache::new(2);
        assert_eq!(cache.insert(&[1], 1), None);
        assert_eq!(cache.insert(&[2], 2), None);
        assert_eq!(cache.longest_prefix(&[1]), Some((1, 1)));
        assert_eq!(cache.insert(&[1], 3), Some(1));
        assert_eq!(cache.insert(&[3], 4), Some(2));
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.longest_prefix(&[1]), Some((1, 3)));
        assert_eq!(cache.longest_prefix(&[2]), None);
    }

    #[test]
    fn radix_prefix_zero_capacity_and_clear() {
        let mut cache = RadixPrefixCache::new(0);
        assert_eq!(cache.insert(&[1], 1), None);
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.longest_prefix(&[1]), None);

        let mut cache = RadixPrefixCache::new(2);
        assert_eq!(cache.insert(&[], 7), None);
        assert_eq!(cache.longest_prefix(&[]), Some((0, 7)));
        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.longest_prefix(&[]), None);
    }

    proptest! {
        #[test]
        fn kv_append_rollback_free_invariants(
            tokens in proptest::collection::vec(0i32..100, 0..40),
            rollback in 0usize..41,
        ) {
            let mut cache = PagedKvCache::new(4, 16).expect("valid cache");
            let seq = cache.allocate(&request(), tokens.len()).expect("allocation");
            cache.append(seq, &tokens).expect("append");
            prop_assert_eq!(cache.snapshot(seq).expect("snapshot"), tokens.clone());
            prop_assert_eq!(
                cache.blocks_used(),
                tokens.len().div_ceil(4)
            );

            let amount = rollback.min(tokens.len());
            cache.rollback(seq, amount).expect("rollback");
            let expected = tokens[..tokens.len() - amount].to_vec();
            prop_assert_eq!(cache.snapshot(seq).expect("snapshot"), expected.clone());
            prop_assert_eq!(cache.blocks_used(), expected.len().div_ceil(4));

            cache.free(seq).expect("free");
            prop_assert_eq!(cache.blocks_used(), 0);
        }

        #[test]
        fn radix_prefix_returns_observed_values(
            first in proptest::collection::vec(0i32..8, 1..8),
            second in proptest::collection::vec(0i32..8, 1..8),
            first_value in any::<u64>(),
            second_value in any::<u64>(),
        ) {
            let same_key = first == second;
            let mut cache = RadixPrefixCache::new(2);
            prop_assert_eq!(cache.insert(&first, first_value), None);
            let displaced = if same_key { Some(first_value) } else { None };
            prop_assert_eq!(cache.insert(&second, second_value), displaced);
            prop_assert!(cache.len() <= 2);

            let first_match = first.len();
            let second_match = second
                .iter()
                .zip(first.iter())
                .take_while(|(left, right)| left == right)
                .count();
            let expected_length = first_match.max(second_match);
            let mut allowed_values = Vec::new();
            if first_match == expected_length {
                let value = if same_key { second_value } else { first_value };
                allowed_values.push(value);
            }
            if !same_key && second_match == expected_length {
                allowed_values.push(second_value);
            }

            let (matched, value) = cache
                .longest_prefix(&first)
                .expect("inserted prefix is present");
            prop_assert_eq!(matched, expected_length);
            prop_assert!(allowed_values.contains(&value));
        }
    }
}
