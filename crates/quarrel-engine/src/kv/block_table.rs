//! Explicit block tables with reference-counted prefix sharing.
//!
//! [`BlockTableKvCache`] keeps a fixed pool of token blocks. Each sequence owns
//! a table of block indices plus a logical length, so callers can inspect the
//! physical layout with [`BlockTableKvCache::block_table`]. Prefixes created by
//! [`BlockTableKvCache::share_prefix`] are immutable: writes to a shared block
//! copy it into a private block first, and a block is returned to the free list
//! only when the last table referencing it is rolled back or freed.

use crate::EngineError;
use quarrel_core::RequestId;
use std::collections::{HashMap, HashSet};

/// A bounded KV cache with explicit per-sequence block tables.
///
/// Blocks hold exactly `block_size` tokens. A sequence may reserve blocks with
/// [`BlockTableKvCache::allocate`], share a prefix with another sequence, and
/// grow with [`BlockTableKvCache::append`]. Shared blocks are copy-on-write, so
/// neither owner can observe the other's mutations.
#[derive(Debug)]
pub struct BlockTableKvCache {
    block_size: usize,
    capacity: usize,
    blocks: Vec<Option<Vec<i32>>>,
    refcounts: Vec<usize>,
    free_blocks: Vec<usize>,
    sequences: HashMap<u64, BlockTableSequence>,
    next_sequence: u64,
}

#[derive(Debug)]
struct BlockTableSequence {
    table: Vec<usize>,
    length: usize,
}

impl BlockTableKvCache {
    /// Creates an empty cache with `capacity` blocks of `block_size` tokens.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::InvalidBlockSize`] if `block_size` is zero and
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
            refcounts: vec![0; capacity],
            free_blocks: (0..capacity).rev().collect(),
            sequences: HashMap::new(),
            next_sequence: 1,
        })
    }

    /// Reserves `token_count` tokens of private capacity and returns a sequence.
    ///
    /// The new sequence starts empty; call [`BlockTableKvCache::append`] to fill
    /// it. The `request_id` is not stored, but is accepted for symmetry with
    /// [`crate::kv::PagedKvCache::allocate`].
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::CapacityExceeded`] when the cache cannot reserve
    /// enough blocks, or [`EngineError::CacheInvariant`] if the sequence id
    /// space is exhausted.
    pub fn allocate(
        &mut self,
        _request_id: &RequestId,
        token_count: usize,
    ) -> Result<u64, EngineError> {
        let required = token_count.div_ceil(self.block_size);
        if required > self.capacity || required > self.free_blocks.len() {
            return Err(self.capacity_error(token_count));
        }

        let seq = self.next_sequence;
        self.next_sequence = seq.checked_add(1).ok_or(EngineError::CacheInvariant {
            reason: "sequence id space exhausted",
        })?;

        let mut table = Vec::with_capacity(required);
        for _ in 0..required {
            let index = self.pop_free_block()?;
            table.push(index);
        }
        self.sequences
            .insert(seq, BlockTableSequence { table, length: 0 });
        Ok(seq)
    }

    /// Creates a new sequence that shares the first `tokens` tokens of `source`.
    ///
    /// The shared blocks are immutable and reference-counted. Writing to a
    /// shared block in either sequence transparently copies it first; see
    /// [`BlockTableKvCache::append`].
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::UnknownSequence`] for an unknown source,
    /// [`EngineError::InvalidPrefixLength`] when `tokens` exceeds the source
    /// length, [`EngineError::CapacityExceeded`] when free blocks are lacking,
    /// and [`EngineError::CacheInvariant`] for internal block-table corruption.
    pub fn share_prefix(&mut self, source: u64, tokens: usize) -> Result<u64, EngineError> {
        let (source_table, source_length) = {
            let sequence = self
                .sequences
                .get(&source)
                .ok_or(EngineError::UnknownSequence { seq: source })?;
            (sequence.table.clone(), sequence.length)
        };
        if tokens > source_length {
            return Err(EngineError::InvalidPrefixLength {
                requested: tokens,
                available: source_length,
            });
        }

        let required = tokens.div_ceil(self.block_size);
        if required > self.capacity || required > self.free_blocks.len() {
            return Err(self.capacity_error(tokens));
        }
        let prefix = source_table
            .get(..required)
            .ok_or(EngineError::CacheInvariant {
                reason: "source table is shorter than its logical length",
            })?;
        self.validate_table(prefix)?;

        let seq = self.next_sequence;
        self.next_sequence = seq.checked_add(1).ok_or(EngineError::CacheInvariant {
            reason: "sequence id space exhausted",
        })?;

        let mut table = Vec::with_capacity(required);
        for &index in prefix {
            let incremented = self
                .refcounts
                .get(index)
                .copied()
                .ok_or(EngineError::CacheInvariant {
                    reason: "block index outside refcount table",
                })?
                .checked_add(1)
                .ok_or(EngineError::CacheInvariant {
                    reason: "block refcount overflow",
                })?;
            let slot = self
                .refcounts
                .get_mut(index)
                .ok_or(EngineError::CacheInvariant {
                    reason: "block index outside refcount table",
                })?;
            *slot = incremented;
            table.push(index);
        }
        self.sequences.insert(
            seq,
            BlockTableSequence {
                table,
                length: tokens,
            },
        );
        Ok(seq)
    }

    /// Appends `tokens` to the logical stream of `seq`.
    ///
    /// Blocks already owned exclusively are written in place. Shared blocks are
    /// copied into a private block before the write. All capacity checks happen
    /// before any mutation, so a failed append leaves the cache unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::UnknownSequence`] for an unknown id,
    /// [`EngineError::CapacityExceeded`] when extra or copy-on-write blocks do
    /// not fit, and [`EngineError::CacheInvariant`] for internal corruption.
    pub fn append(&mut self, seq: u64, tokens: &[i32]) -> Result<(), EngineError> {
        if tokens.is_empty() {
            if !self.sequences.contains_key(&seq) {
                return Err(EngineError::UnknownSequence { seq });
            }
            return Ok(());
        }

        let table = self
            .sequences
            .get(&seq)
            .ok_or(EngineError::UnknownSequence { seq })?
            .table
            .clone();
        let length = self
            .sequences
            .get(&seq)
            .ok_or(EngineError::UnknownSequence { seq })?
            .length;
        self.validate_table(&table)?;

        let final_length = length
            .checked_add(tokens.len())
            .ok_or(EngineError::CacheInvariant {
                reason: "sequence length overflow",
            })?;
        let required_blocks = final_length.div_ceil(self.block_size);
        if required_blocks > self.capacity {
            return Err(self.capacity_error(final_length));
        }

        let first_written = length / self.block_size;
        let last_written = (final_length - 1) / self.block_size;
        let mut copy_on_write = Vec::new();
        for position in first_written..=last_written {
            let Some(&index) = table.get(position) else {
                continue;
            };
            let refcount =
                self.refcounts
                    .get(index)
                    .copied()
                    .ok_or(EngineError::CacheInvariant {
                        reason: "block index outside refcount table",
                    })?;
            if refcount == 0 {
                return Err(EngineError::CacheInvariant {
                    reason: "owned block has no references",
                });
            }
            if refcount > 1 {
                copy_on_write.push(position);
            }
        }

        let extra_blocks = required_blocks.saturating_sub(table.len());
        let needed =
            extra_blocks
                .checked_add(copy_on_write.len())
                .ok_or(EngineError::CacheInvariant {
                    reason: "block request overflow",
                })?;
        if needed > self.free_blocks.len() {
            return Err(self.capacity_error(final_length));
        }

        for _ in 0..extra_blocks {
            let index = self.pop_free_block()?;
            self.sequences
                .get_mut(&seq)
                .ok_or(EngineError::UnknownSequence { seq })?
                .table
                .push(index);
        }

        for position in copy_on_write {
            let old_index = *self
                .sequences
                .get(&seq)
                .ok_or(EngineError::UnknownSequence { seq })?
                .table
                .get(position)
                .ok_or(EngineError::CacheInvariant {
                    reason: "copy-on-write position outside table",
                })?;
            let old_block = self
                .blocks
                .get(old_index)
                .and_then(Option::as_ref)
                .cloned()
                .ok_or(EngineError::CacheInvariant {
                    reason: "shared block is not allocated",
                })?;
            let new_index = self.pop_free_block()?;
            let copied = self
                .blocks
                .get_mut(new_index)
                .ok_or(EngineError::CacheInvariant {
                    reason: "new block index outside block pool",
                })?;
            *copied = Some(old_block);
            let slot = self
                .sequences
                .get_mut(&seq)
                .ok_or(EngineError::UnknownSequence { seq })?
                .table
                .get_mut(position)
                .ok_or(EngineError::CacheInvariant {
                    reason: "copy-on-write position outside table",
                })?;
            *slot = new_index;
            self.release_block(old_index)?;
        }

        for (offset, &token) in tokens.iter().enumerate() {
            let position = length
                .checked_add(offset)
                .ok_or(EngineError::CacheInvariant {
                    reason: "sequence length overflow",
                })?;
            let slot = position / self.block_size;
            let within_block = position % self.block_size;
            let block_index = *self
                .sequences
                .get(&seq)
                .ok_or(EngineError::UnknownSequence { seq })?
                .table
                .get(slot)
                .ok_or(EngineError::CacheInvariant {
                    reason: "token position outside block table",
                })?;
            let block = self
                .blocks
                .get_mut(block_index)
                .and_then(Option::as_mut)
                .ok_or(EngineError::CacheInvariant {
                    reason: "owned block is not allocated",
                })?;
            let cell = block
                .get_mut(within_block)
                .ok_or(EngineError::CacheInvariant {
                    reason: "token offset outside block",
                })?;
            *cell = token;
        }
        self.sequences
            .get_mut(&seq)
            .ok_or(EngineError::UnknownSequence { seq })?
            .length = final_length;
        Ok(())
    }

    /// Truncates `seq` to `length - token_count` tokens and releases unused blocks.
    ///
    /// Only blocks that are no longer in the sequence's table are released. A
    /// shared prefix block stays alive while any other table still references
    /// it. Rolling back more tokens than the sequence holds is an error.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::UnknownSequence`] for an unknown id,
    /// [`EngineError::InvalidRollback`] for an oversized rollback and
    /// [`EngineError::CacheInvariant`] for internal corruption.
    pub fn rollback(&mut self, seq: u64, token_count: usize) -> Result<(), EngineError> {
        let (length, table) = {
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
            (sequence.length, sequence.table.clone())
        };
        self.validate_table(&table)?;

        let remaining = length - token_count;
        let keep_blocks = remaining.div_ceil(self.block_size);
        let removed = table
            .get(keep_blocks..)
            .ok_or(EngineError::CacheInvariant {
                reason: "logical length exceeds block table",
            })?
            .to_vec();

        let sequence = self
            .sequences
            .get_mut(&seq)
            .ok_or(EngineError::UnknownSequence { seq })?;
        sequence.length = remaining;
        sequence.table.truncate(keep_blocks);
        for index in removed {
            self.release_block(index)?;
        }
        Ok(())
    }

    /// Frees `seq` and releases each of its blocks once.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::UnknownSequence`] if the id is unknown or already
    /// freed, and [`EngineError::CacheInvariant`] for internal corruption.
    pub fn free(&mut self, seq: u64) -> Result<(), EngineError> {
        let table = self
            .sequences
            .get(&seq)
            .ok_or(EngineError::UnknownSequence { seq })?
            .table
            .clone();
        self.validate_table(&table)?;
        self.sequences.remove(&seq);
        for index in table {
            self.release_block(index)?;
        }
        Ok(())
    }

    /// Returns the physical block indices owned by `seq`, in logical order.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::UnknownSequence`] if the id is unknown.
    pub fn block_table(&self, seq: u64) -> Result<&[usize], EngineError> {
        let sequence = self
            .sequences
            .get(&seq)
            .ok_or(EngineError::UnknownSequence { seq })?;
        Ok(sequence.table.as_slice())
    }

    /// Returns the reference count of a block, or `None` for an invalid index.
    #[must_use]
    pub fn block_refcount(&self, block: usize) -> Option<usize> {
        self.refcounts.get(block).copied()
    }

    /// Returns the number of blocks currently shared by two or more sequences.
    #[must_use]
    pub fn shared_blocks(&self) -> usize {
        self.refcounts.iter().filter(|&&count| count > 1).count()
    }

    /// Returns the number of blocks not owned by any sequence.
    #[must_use]
    pub fn free_blocks(&self) -> usize {
        self.free_blocks.len()
    }

    /// Returns the number of blocks currently allocated to sequences.
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
        let (length, table) = {
            let sequence = self
                .sequences
                .get(&seq)
                .ok_or(EngineError::UnknownSequence { seq })?;
            (sequence.length, sequence.table.clone())
        };
        self.validate_table(&table)?;

        let mut tokens = Vec::with_capacity(length);
        for &block_index in &table {
            let remaining = length.saturating_sub(tokens.len());
            if remaining == 0 {
                break;
            }
            let block = self
                .blocks
                .get(block_index)
                .and_then(Option::as_ref)
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

    fn pop_free_block(&mut self) -> Result<usize, EngineError> {
        let index = self.free_blocks.pop().ok_or(EngineError::CacheInvariant {
            reason: "free list disagrees with block count",
        })?;
        if self.refcounts.get(index) != Some(&0) {
            return Err(EngineError::CacheInvariant {
                reason: "free block is still referenced",
            });
        }
        let block = self
            .blocks
            .get_mut(index)
            .ok_or(EngineError::CacheInvariant {
                reason: "free block index outside block pool",
            })?;
        if block.is_some() {
            return Err(EngineError::CacheInvariant {
                reason: "free block is still allocated",
            });
        }
        *block = Some(vec![0; self.block_size]);
        let refcount = self
            .refcounts
            .get_mut(index)
            .ok_or(EngineError::CacheInvariant {
                reason: "free block index outside refcount table",
            })?;
        *refcount = 1;
        Ok(index)
    }

    fn release_block(&mut self, index: usize) -> Result<(), EngineError> {
        let remaining = self
            .refcounts
            .get(index)
            .copied()
            .ok_or(EngineError::CacheInvariant {
                reason: "block index outside refcount table",
            })?;
        if remaining == 0 {
            return Err(EngineError::CacheInvariant {
                reason: "block refcount underflow",
            });
        }
        if remaining > 1 {
            let refcount = self
                .refcounts
                .get_mut(index)
                .ok_or(EngineError::CacheInvariant {
                    reason: "block index outside refcount table",
                })?;
            *refcount = remaining - 1;
            return Ok(());
        }
        let block = self
            .blocks
            .get_mut(index)
            .ok_or(EngineError::CacheInvariant {
                reason: "block index outside block pool",
            })?;
        if block.is_none() {
            return Err(EngineError::CacheInvariant {
                reason: "last reference held an unallocated block",
            });
        }
        *block = None;
        let refcount = self
            .refcounts
            .get_mut(index)
            .ok_or(EngineError::CacheInvariant {
                reason: "block index outside refcount table",
            })?;
        *refcount = 0;
        self.free_blocks.push(index);
        Ok(())
    }

    fn validate_table(&self, table: &[usize]) -> Result<(), EngineError> {
        let mut seen = HashSet::with_capacity(table.len());
        for &index in table {
            if !seen.insert(index) {
                return Err(EngineError::CacheInvariant {
                    reason: "duplicate block in sequence table",
                });
            }
            let block = self.blocks.get(index).and_then(Option::as_ref).ok_or(
                EngineError::CacheInvariant {
                    reason: "table references an unallocated block",
                },
            )?;
            if block.len() != self.block_size {
                return Err(EngineError::CacheInvariant {
                    reason: "allocated block has the wrong size",
                });
            }
            let refcount =
                self.refcounts
                    .get(index)
                    .copied()
                    .ok_or(EngineError::CacheInvariant {
                        reason: "block index outside refcount table",
                    })?;
            if refcount == 0 {
                return Err(EngineError::CacheInvariant {
                    reason: "table references a block with no owners",
                });
            }
        }
        Ok(())
    }

    fn capacity_error(&self, requested: usize) -> EngineError {
        EngineError::CapacityExceeded {
            capacity: self.capacity.saturating_mul(self.block_size),
            requested,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::BlockTableKvCache;
    use crate::EngineError;
    use quarrel_core::RequestId;

    fn request(id: &str) -> RequestId {
        RequestId::new(id).expect("non-empty request id")
    }

    #[test]
    fn sequences_have_independent_explicit_block_tables() {
        let mut cache = BlockTableKvCache::new(2, 4).expect("valid cache");
        let first = cache.allocate(&request("first"), 4).expect("allocation");
        let second = cache.allocate(&request("second"), 4).expect("allocation");
        assert_ne!(first, second);
        assert_eq!(cache.block_table(first).expect("table").len(), 2);
        assert_ne!(
            cache.block_table(first).expect("table"),
            cache.block_table(second).expect("table")
        );

        cache.append(first, &[1, 2, 3, 4]).expect("append");
        cache.append(second, &[9, 8, 7, 6]).expect("append");
        assert_eq!(cache.snapshot(first), Ok(vec![1, 2, 3, 4]));
        assert_eq!(cache.snapshot(second), Ok(vec![9, 8, 7, 6]));

        cache.rollback(first, 3).expect("rollback");
        assert_eq!(cache.snapshot(first), Ok(vec![1]));
        assert_eq!(cache.snapshot(second), Ok(vec![9, 8, 7, 6]));
        assert_eq!(cache.block_table(first).expect("table").len(), 1);
    }

    #[test]
    fn rollback_truncates_logical_length_without_freeing_shared_prefix() {
        let mut cache = BlockTableKvCache::new(2, 6).expect("valid cache");
        let parent = cache.allocate(&request("parent"), 4).expect("allocation");
        cache.append(parent, &[1, 2, 3, 4]).expect("append");
        let child = cache.share_prefix(parent, 3).expect("shared prefix");
        assert_eq!(
            &cache.block_table(child).expect("child table")[..2],
            cache.block_table(parent).expect("parent table")
        );

        cache.append(child, &[5]).expect("copy-on-write append");
        assert_ne!(cache.block_table(child), cache.block_table(parent));
        assert_eq!(cache.snapshot(child), Ok(vec![1, 2, 3, 5]));
        assert_eq!(cache.snapshot(parent), Ok(vec![1, 2, 3, 4]));
        assert_eq!(cache.shared_blocks(), 1);

        cache
            .rollback(child, 2)
            .expect("rollback into shared prefix");
        assert_eq!(cache.snapshot(child), Ok(vec![1, 2]));
        assert_eq!(cache.snapshot(parent), Ok(vec![1, 2, 3, 4]));
        assert_eq!(cache.shared_blocks(), 1);
        assert_eq!(cache.block_refcount(0), Some(2));

        cache.free(child).expect("free child");
        assert_eq!(cache.block_refcount(0), Some(1));
        assert_eq!(cache.snapshot(parent), Ok(vec![1, 2, 3, 4]));
        cache.free(parent).expect("free parent");
        assert_eq!(cache.blocks_used(), 0);
    }

    #[test]
    fn shared_blocks_are_freed_only_when_the_last_owner_releases_them() {
        let mut cache = BlockTableKvCache::new(2, 4).expect("valid cache");
        let first = cache.allocate(&request("first"), 4).expect("allocation");
        cache.append(first, &[10, 11, 12, 13]).expect("append");
        let second = cache.share_prefix(first, 4).expect("shared prefix");
        assert_eq!(cache.blocks_used(), 2);
        assert_eq!(cache.shared_blocks(), 2);

        cache.free(first).expect("free first owner");
        assert_eq!(cache.snapshot(second), Ok(vec![10, 11, 12, 13]));
        assert_eq!(cache.blocks_used(), 2);
        assert_eq!(cache.shared_blocks(), 0);

        cache.free(second).expect("free last owner");
        assert_eq!(cache.blocks_used(), 0);

        let reused = cache.allocate(&request("reused"), 4).expect("reuse blocks");
        assert_eq!(cache.snapshot(reused), Ok(Vec::new()));
        cache
            .append(reused, &[7, 7, 7, 7])
            .expect("append after reuse");
        assert_eq!(cache.snapshot(reused), Ok(vec![7, 7, 7, 7]));
        assert!(matches!(
            cache.snapshot(first),
            Err(EngineError::UnknownSequence { .. })
        ));
    }

    #[test]
    fn capacity_exhaustion_is_typed_and_does_not_mutate_sequences() {
        let mut cache = BlockTableKvCache::new(2, 2).expect("valid cache");
        let first = cache.allocate(&request("first"), 4).expect("allocation");
        assert!(matches!(
            cache.allocate(&request("second"), 1),
            Err(EngineError::CapacityExceeded { .. })
        ));
        assert!(matches!(
            cache.append(first, &[1, 2, 3, 4, 5]),
            Err(EngineError::CapacityExceeded { .. })
        ));
        assert!(matches!(
            cache.share_prefix(first, 5),
            Err(EngineError::InvalidPrefixLength {
                requested: 5,
                available: 0
            })
        ));
        cache
            .append(first, &[1, 2, 3, 4])
            .expect("fill reserved blocks");
        assert!(matches!(
            cache.share_prefix(first, 4),
            Err(EngineError::CapacityExceeded { .. })
        ));
        assert_eq!(cache.snapshot(first), Ok(vec![1, 2, 3, 4]));
        assert_eq!(cache.blocks_used(), 2);
    }

    #[test]
    fn invalid_and_double_free_are_reported_without_double_release() {
        let mut cache = BlockTableKvCache::new(2, 2).expect("valid cache");
        let seq = cache.allocate(&request("only"), 2).expect("allocation");
        cache.append(seq, &[1, 2]).expect("append");
        cache.free(seq).expect("free");
        assert_eq!(cache.free_blocks(), 2);
        assert_eq!(cache.block_refcount(0), Some(0));

        assert!(matches!(
            cache.free(seq),
            Err(EngineError::UnknownSequence { seq: returned }) if returned == seq
        ));
        assert!(matches!(
            cache.rollback(seq, 0),
            Err(EngineError::UnknownSequence { .. })
        ));
        assert!(matches!(
            cache.append(seq, &[3]),
            Err(EngineError::UnknownSequence { .. })
        ));
        assert!(matches!(
            cache.block_table(seq),
            Err(EngineError::UnknownSequence { .. })
        ));
        assert_eq!(cache.free_blocks(), 2);
    }

    #[test]
    fn new_validates_geometry() {
        assert!(matches!(
            BlockTableKvCache::new(0, 1),
            Err(EngineError::InvalidBlockSize)
        ));
        assert!(matches!(
            BlockTableKvCache::new(1, 0),
            Err(EngineError::CacheInvariant { .. })
        ));
    }
}
