//! Compressed radix tree with LRU eviction and borrowed prefix handles.
//!
//! The tree stores one terminal entry per inserted token sequence. Edges hold
//! compressed token segments so branching sequences share their common prefix
//! in the index. Lookups return either the legacy `(matched_len, u64)` pair or
//! a [`PrefixRef`] borrowing the stored key and payload without copying them.

/// A borrowed view of the best matching stored prefix.
///
/// The slices are owned by the cache and remain valid until the cache is
/// mutably borrowed again. No token or payload data is cloned on lookup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefixRef<'a> {
    /// Number of leading query tokens matched by the stored key.
    pub len: usize,
    /// Borrowed prefix of the stored key, never longer than `len`.
    pub tokens: &'a [i32],
    /// Borrowed payload stored with that key.
    pub value: &'a u64,
}

#[derive(Debug)]
struct RadixEntry {
    tokens: Vec<i32>,
    node: usize,
    value: u64,
    last_used: u64,
}

#[derive(Debug, Default)]
struct RadixNode {
    entry: Option<usize>,
    children: Vec<RadixEdge>,
}

#[derive(Debug)]
struct RadixEdge {
    label: Vec<i32>,
    child: usize,
}

/// An LRU cache mapping token sequences to `u64` payloads in a radix tree.
///
/// Inserting an identical sequence replaces its payload and refreshes its
/// recency. When the cache is full, the least recently used entry is evicted;
/// either way `insert` returns the displaced payload, if any. Prefix lookups
/// also refresh recency, so a popular prefix is less likely to be evicted.
#[derive(Debug)]
pub struct RadixPrefixCache {
    capacity: usize,
    nodes: Vec<Option<RadixNode>>,
    free_nodes: Vec<usize>,
    entries: Vec<Option<RadixEntry>>,
    free_entries: Vec<usize>,
    clock: u64,
    len: usize,
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
            nodes: vec![Some(RadixNode::default())],
            free_nodes: Vec::new(),
            entries: Vec::new(),
            free_entries: Vec::new(),
            clock: 0,
            len: 0,
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

        if let Some(entry) = self.find_entry(tokens) {
            let displaced = match self.entries.get_mut(entry).and_then(Option::as_mut) {
                Some(previous) => {
                    let old = previous.value;
                    previous.value = value;
                    Some(old)
                }
                None => None,
            };
            if displaced.is_some() {
                self.touch_entry(entry);
            }
            return displaced;
        }

        let displaced = if self.len >= self.capacity {
            self.evict_lru()
        } else {
            None
        };
        if self.len >= self.capacity && displaced.is_none() {
            return displaced;
        }

        let last_used = self.tick();
        let entry = self.alloc_entry(RadixEntry {
            tokens: tokens.to_vec(),
            node: 0,
            value,
            last_used,
        });
        let Some(node) = self.insert_path(entry, tokens) else {
            self.remove_entry(entry);
            self.len = self.len.saturating_sub(1);
            return displaced;
        };
        if let Some(stored) = self.entries.get_mut(entry).and_then(Option::as_mut) {
            stored.node = node;
        }
        self.len = self.len.saturating_add(1);
        displaced
    }

    /// Returns the longest stored prefix of `tokens` and its payload.
    ///
    /// A matching empty prefix counts when an empty key was inserted. Ties are
    /// broken in favour of the most recently used entry, which is then touched.
    pub fn longest_prefix(&mut self, tokens: &[i32]) -> Option<(usize, u64)> {
        let (length, entry) = self.find_best(tokens)?;
        let value = self
            .entries
            .get(entry)
            .and_then(Option::as_ref)
            .map(|stored| stored.value)?;
        self.touch_entry(entry);
        Some((length, value))
    }

    /// Returns a borrowed view of the longest stored prefix of `tokens`.
    ///
    /// This has the same matching and recency rules as
    /// [`RadixPrefixCache::longest_prefix`], but returns borrowed token and
    /// payload references instead of copying the payload value. The returned
    /// references borrow the cache, so the cache cannot be mutated while the
    /// handle is alive.
    pub fn longest_prefix_ref(&mut self, tokens: &[i32]) -> Option<PrefixRef<'_>> {
        let (length, entry) = self.find_best(tokens)?;
        self.touch_entry(entry);
        let stored = self.entries.get(entry)?.as_ref()?;
        let matched = stored.tokens.get(..length)?;
        Some(PrefixRef {
            len: length,
            tokens: matched,
            value: &stored.value,
        })
    }

    /// Returns the number of entries currently stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` when no entries are stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Removes all entries without changing the capacity.
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.nodes.push(Some(RadixNode::default()));
        self.free_nodes.clear();
        self.entries.clear();
        self.free_entries.clear();
        self.clock = 0;
        self.len = 0;
    }

    fn find_entry(&self, tokens: &[i32]) -> Option<usize> {
        let mut node = 0usize;
        let mut position = 0usize;
        loop {
            let current = self.node(node)?;
            if position == tokens.len() {
                return current.entry;
            }
            let next = tokens.get(position)?;
            let edge = current
                .children
                .iter()
                .find(|edge| edge.label.first() == Some(next))?;
            let mut offset = 0usize;
            while offset < edge.label.len() {
                if edge.label.get(offset) != tokens.get(position + offset) {
                    return None;
                }
                offset += 1;
            }
            position += offset;
            node = edge.child;
        }
    }

    fn find_best(&self, tokens: &[i32]) -> Option<(usize, usize)> {
        if tokens.is_empty() {
            let root = self.node(0)?;
            return root.entry.map(|entry| (0, entry));
        }

        let mut best: Option<(usize, u64, usize)> = None;
        let mut consider = |length: usize, entry: usize| {
            let Some(recency) = self
                .entries
                .get(entry)
                .and_then(Option::as_ref)
                .map(|stored| stored.last_used)
            else {
                return;
            };
            let better = match best {
                Some((best_length, best_recency, _)) => {
                    length > best_length || (length == best_length && recency > best_recency)
                }
                None => true,
            };
            if better {
                best = Some((length, recency, entry));
            }
        };

        let mut node = 0usize;
        let mut position = 0usize;
        while let Some(current) = self.node(node) {
            if let Some(entry) = current.entry
                && position > 0
            {
                consider(position, entry);
            }
            if position == tokens.len() {
                if let Some((entry, _)) = self.most_recent_terminal(node) {
                    consider(position, entry);
                }
                break;
            }
            let next = match tokens.get(position) {
                Some(next) => *next,
                None => break,
            };
            let edge = match current
                .children
                .iter()
                .find(|edge| edge.label.first() == Some(&next))
            {
                Some(edge) => edge,
                None => {
                    if position > 0
                        && let Some((entry, _)) = self.most_recent_terminal(node)
                    {
                        consider(position, entry);
                    }
                    break;
                }
            };
            let mut offset = 0usize;
            while offset < edge.label.len() {
                let expected = match edge.label.get(offset) {
                    Some(expected) => *expected,
                    None => break,
                };
                let actual = match tokens.get(position + offset) {
                    Some(actual) => *actual,
                    None => break,
                };
                if expected != actual {
                    break;
                }
                offset += 1;
            }
            let matched = position + offset;
            if offset < edge.label.len() {
                if matched > 0
                    && let Some((entry, _)) = self.most_recent_terminal(edge.child)
                {
                    consider(matched, entry);
                }
                break;
            }
            position = matched;
            node = edge.child;
        }
        best.map(|(length, _, entry)| (length, entry))
    }

    fn most_recent_terminal(&self, root: usize) -> Option<(usize, u64)> {
        let mut stack = vec![root];
        let mut best: Option<(usize, u64)> = None;
        while let Some(node) = stack.pop() {
            let Some(current) = self.node(node) else {
                continue;
            };
            if let Some(entry) = current.entry
                && let Some(stored) = self.entries.get(entry).and_then(Option::as_ref)
                && best.is_none_or(|(_, recency)| stored.last_used > recency)
            {
                best = Some((entry, stored.last_used));
            }
            for edge in &current.children {
                stack.push(edge.child);
            }
        }
        best
    }

    fn insert_path(&mut self, entry: usize, tokens: &[i32]) -> Option<usize> {
        if tokens.is_empty() {
            let root = self.node_mut(0)?;
            root.entry = Some(entry);
            return Some(0);
        }

        let mut node = 0usize;
        let mut position = 0usize;
        loop {
            if position == tokens.len() {
                let current = self.node_mut(node)?;
                current.entry = Some(entry);
                return Some(node);
            }
            let next = *tokens.get(position)?;
            let edge_index = self
                .node(node)?
                .children
                .iter()
                .position(|edge| edge.label.first() == Some(&next));

            let Some(edge_index) = edge_index else {
                let child = self.alloc_node(RadixNode::default());
                let label = tokens.get(position..)?.to_vec();
                let current = self.node_mut(node)?;
                current.children.push(RadixEdge { label, child });
                let leaf = self.node_mut(child)?;
                leaf.entry = Some(entry);
                return Some(child);
            };

            let edge = self
                .node(node)
                .and_then(|current| current.children.get(edge_index))?;
            let (label, old_child) = (edge.label.clone(), edge.child);
            let mut common = 0usize;
            while common < label.len() {
                let expected = *label.get(common)?;
                let actual = match tokens.get(position + common) {
                    Some(actual) => *actual,
                    None => break,
                };
                if expected != actual {
                    break;
                }
                common += 1;
            }
            if common == label.len() {
                position += common;
                node = old_child;
                continue;
            }

            let split = self.alloc_node(RadixNode::default());
            let prefix = label.get(..common)?.to_vec();
            let suffix = label.get(common..)?.to_vec();
            let split_node = self.node_mut(split)?;
            split_node.children.push(RadixEdge {
                label: suffix,
                child: old_child,
            });
            let current = self.node_mut(node)?;
            let slot = current.children.get_mut(edge_index)?;
            *slot = RadixEdge {
                label: prefix,
                child: split,
            };
            position += common;
            node = split;
        }
    }

    fn evict_lru(&mut self) -> Option<u64> {
        let victim = self
            .entries
            .iter()
            .enumerate()
            .filter_map(|(id, slot)| slot.as_ref().map(|stored| (id, stored.last_used)))
            .min_by_key(|(_, recency)| *recency)
            .map(|(id, _)| id)?;
        let stored = self.entries.get_mut(victim)?.take()?;
        self.free_entries.push(victim);
        self.len = self.len.saturating_sub(1);
        if let Some(node) = self.node_mut(stored.node) {
            node.entry = None;
        }
        self.prune_node(stored.node);
        Some(stored.value)
    }

    fn prune_node(&mut self, mut node: usize) {
        if node == 0 {
            return;
        }
        loop {
            let removable = self
                .node(node)
                .is_some_and(|current| current.entry.is_none() && current.children.is_empty());
            if !removable {
                return;
            }
            let Some((parent, edge_index)) = self.find_parent(node) else {
                return;
            };
            let Some(parent_node) = self.node_mut(parent) else {
                return;
            };
            if edge_index >= parent_node.children.len() {
                return;
            }
            parent_node.children.remove(edge_index);
            if let Some(slot) = self.nodes.get_mut(node) {
                *slot = None;
            }
            self.free_nodes.push(node);
            node = parent;
        }
    }

    fn find_parent(&self, child: usize) -> Option<(usize, usize)> {
        for (id, slot) in self.nodes.iter().enumerate() {
            if let Some(node) = slot.as_ref()
                && let Some(position) = node.children.iter().position(|edge| edge.child == child)
            {
                return Some((id, position));
            }
        }
        None
    }

    fn alloc_node(&mut self, node: RadixNode) -> usize {
        if let Some(id) = self.free_nodes.pop()
            && let Some(slot) = self.nodes.get_mut(id)
        {
            *slot = Some(node);
            return id;
        }
        self.nodes.push(Some(node));
        self.nodes.len().saturating_sub(1)
    }

    fn alloc_entry(&mut self, entry: RadixEntry) -> usize {
        if let Some(id) = self.free_entries.pop()
            && let Some(slot) = self.entries.get_mut(id)
        {
            *slot = Some(entry);
            return id;
        }
        self.entries.push(Some(entry));
        self.entries.len().saturating_sub(1)
    }

    fn remove_entry(&mut self, entry: usize) {
        if let Some(slot) = self.entries.get_mut(entry)
            && slot.take().is_some()
        {
            self.free_entries.push(entry);
        }
    }

    fn touch_entry(&mut self, entry: usize) {
        let recency = self.tick();
        if let Some(stored) = self.entries.get_mut(entry).and_then(Option::as_mut) {
            stored.last_used = recency;
        }
    }

    fn tick(&mut self) -> u64 {
        self.clock = self.clock.wrapping_add(1);
        self.clock
    }

    fn node(&self, id: usize) -> Option<&RadixNode> {
        self.nodes.get(id).and_then(Option::as_ref)
    }

    fn node_mut(&mut self, id: usize) -> Option<&mut RadixNode> {
        self.nodes.get_mut(id).and_then(Option::as_mut)
    }
}

#[cfg(test)]
mod tests {
    use super::{PrefixRef, RadixPrefixCache};
    use proptest::prelude::*;

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

    #[test]
    fn branching_selects_the_deepest_matching_path() {
        let mut cache = RadixPrefixCache::new(8);
        assert_eq!(cache.insert(&[1, 2, 3], 100), None);
        assert_eq!(cache.insert(&[1, 2, 4], 101), None);
        assert_eq!(cache.insert(&[1, 5], 102), None);
        assert_eq!(cache.insert(&[9, 2, 3], 103), None);

        assert_eq!(cache.longest_prefix(&[1, 2, 3, 7]), Some((3, 100)));
        assert_eq!(cache.longest_prefix(&[1, 2, 4, 7]), Some((3, 101)));
        assert_eq!(cache.longest_prefix(&[1, 5, 9]), Some((2, 102)));
        assert_eq!(cache.longest_prefix(&[9, 2, 3]), Some((3, 103)));
        // Ambiguous partial match: `[1, 2]` is the deepest shared prefix.
        assert_eq!(cache.longest_prefix(&[1, 2, 8]), Some((2, 101)));
        // The previous ambiguous lookup refreshed `[1, 2, 4]`, so it wins here.
        assert_eq!(cache.longest_prefix(&[1, 8]), Some((1, 101)));
        assert_eq!(cache.longest_prefix(&[8]), None);
        assert_eq!(cache.len(), 4);
    }

    #[test]
    fn prefix_ties_use_recency_and_touch_the_winner() {
        let mut cache = RadixPrefixCache::new(4);
        assert_eq!(cache.insert(&[1, 2], 10), None);
        assert_eq!(cache.insert(&[1, 3], 11), None);
        assert_eq!(cache.longest_prefix(&[1, 2]), Some((2, 10)));
        // Looking up `[1, 2]` refreshed its entry, so it wins this tie too.
        assert_eq!(cache.longest_prefix(&[1, 9]), Some((1, 10)));
        assert_eq!(cache.longest_prefix(&[1, 9]), Some((1, 10)));

        assert_eq!(cache.insert(&[1, 2, 4], 12), None);
        assert_eq!(cache.longest_prefix(&[1, 2, 8]), Some((2, 12)));
    }

    #[test]
    fn lru_eviction_removes_only_the_least_recent_branch() {
        let mut cache = RadixPrefixCache::new(2);
        assert_eq!(cache.insert(&[1, 2, 3], 1), None);
        assert_eq!(cache.insert(&[1, 2, 4], 2), None);
        assert_eq!(cache.longest_prefix(&[1, 2, 3]), Some((3, 1)));
        assert_eq!(cache.insert(&[1, 2, 5], 3), Some(2));

        assert_eq!(cache.longest_prefix(&[1, 2, 4]), Some((2, 3)));
        assert_eq!(cache.longest_prefix(&[1, 2, 3]), Some((3, 1)));
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn empty_keys_are_ignored_for_non_empty_queries() {
        let mut cache = RadixPrefixCache::new(3);
        assert_eq!(cache.insert(&[], 99), None);
        assert_eq!(cache.insert(&[1, 2], 7), None);
        assert_eq!(cache.longest_prefix(&[]), Some((0, 99)));
        assert_eq!(cache.longest_prefix(&[1]), Some((1, 7)));
        assert_eq!(cache.longest_prefix(&[9]), None);
    }

    #[test]
    fn longest_prefix_ref_borrows_stored_tokens_and_payload() {
        let mut cache = RadixPrefixCache::new(4);
        assert_eq!(cache.insert(&[1, 2, 3], 10), None);
        let (tokens, value): (Vec<i32>, u64) = {
            let matched: PrefixRef<'_> = cache
                .longest_prefix_ref(&[1, 2, 3, 9])
                .expect("borrowed match");
            assert_eq!(matched.len, 3);
            assert_eq!(matched.tokens, &[1, 2, 3]);
            (matched.tokens.to_vec(), *matched.value)
        };
        assert_eq!(tokens, vec![1, 2, 3]);
        assert_eq!(value, 10);

        // Replacing the payload refreshes the same stored entry in place.
        assert_eq!(cache.insert(&[1, 2, 3], 11), Some(10));
        let borrowed = cache
            .longest_prefix_ref(&[1, 2, 3])
            .expect("borrowed match");
        assert_eq!(*borrowed.value, 11);
    }

    proptest! {
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

        #[test]
        fn radix_tree_matches_naive_longest_prefix(
            keys in proptest::collection::vec(proptest::collection::vec(0i32..5, 0..5), 1..6),
            query in proptest::collection::vec(0i32..5, 0..6),
        ) {
            let mut unique = Vec::new();
            for key in keys {
                if !unique.contains(&key) {
                    unique.push(key);
                }
            }
            let mut cache = RadixPrefixCache::new(unique.len());
            for (index, key) in unique.iter().enumerate() {
                prop_assert_eq!(cache.insert(key, index as u64), None);
            }
            let actual = cache.longest_prefix(&query);
            let mut expected: Option<(usize, u64, u64)> = None;
            for (index, key) in unique.iter().enumerate() {
                if key.is_empty() && !query.is_empty() {
                    continue;
                }
                let matched = key
                    .iter()
                    .zip(query.iter())
                    .take_while(|(left, right)| left == right)
                    .count();
                if matched == 0 && !(key.is_empty() && query.is_empty()) {
                    continue;
                }
                let better = match expected {
                    Some((length, recency, _)) => {
                        matched > length || (matched == length && (index as u64) > recency)
                    }
                    None => true,
                };
                if better {
                    expected = Some((matched, index as u64, index as u64));
                }
            }
            match expected {
                Some((length, _, _value)) => {
                    let (matched, found) = actual.expect("expected a match");
                    prop_assert_eq!(matched, length);
                    prop_assert!(found < unique.len() as u64);
                }
                None => prop_assert!(actual.is_none()),
            }
        }
    }
}
