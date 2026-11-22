//! Deterministic iteration-level continuous batching.
//!
//! [`SequenceScheduler`] owns a bounded waiting queue, a bounded set of running
//! sequences and a [`BlockTableKvCache`]. Each `step` advances the next running
//! sequence by at most one token, admits waiting requests when capacity allows
//! and preempts the least recently scheduled sequence when a KV block is
//! needed. Preempted requests keep their sampler state and generated tokens,
//! so they can resume without restarting.
//!
//! The scheduler uses no threads, timers or sleeps. All ordering is FIFO for
//! admission and round-robin for running sequences.

use crate::backend::Backend;
use crate::health::{HealthStatus, audit_logits};
use crate::kv::BlockTableKvCache;
use crate::sampler::Sampler;
use crate::{EngineError, to_usize};
use quarrel_core::{RequestId, SamplingParams, Token};
use std::collections::VecDeque;

/// Bounds and geometry for a [`SequenceScheduler`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SchedulerConfig {
    /// Maximum number of waiting requests, including preempted ones.
    pub max_queue_depth: usize,
    /// Maximum number of sequences producing tokens concurrently.
    pub max_running: usize,
    /// Number of tokens in each KV block.
    pub block_size: usize,
    /// Total number of KV blocks available to the scheduler.
    pub kv_blocks: usize,
}

impl SchedulerConfig {
    /// Creates a scheduler configuration from its four bounds.
    #[must_use]
    pub const fn new(
        max_queue_depth: usize,
        max_running: usize,
        block_size: usize,
        kv_blocks: usize,
    ) -> Self {
        Self {
            max_queue_depth,
            max_running,
            block_size,
            kv_blocks,
        }
    }
}

/// One generation request submitted to a [`SequenceScheduler`].
#[derive(Debug, Clone, PartialEq)]
pub struct GenerationRequest {
    /// Caller-visible request identifier, copied into completion outcomes.
    pub id: RequestId,
    /// Prompt token ids used as the initial context.
    pub prompt: Vec<i32>,
    /// Upper bound on generated tokens, further capped by `params.max_tokens`.
    pub max_tokens: usize,
    /// Sampling parameters used by [`SequenceScheduler::step`].
    pub params: SamplingParams,
    /// Deterministic seed for this request's sampler.
    pub seed: u64,
}

impl GenerationRequest {
    /// Creates a generation request.
    #[must_use]
    pub fn new(
        id: RequestId,
        prompt: Vec<i32>,
        max_tokens: usize,
        params: SamplingParams,
        seed: u64,
    ) -> Self {
        Self {
            id,
            prompt,
            max_tokens,
            params,
            seed,
        }
    }
}

/// Result of one scheduler iteration.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum StepOutcome {
    /// One token was generated; the sequence remains runnable.
    Progress {
        /// Stable scheduler sequence id assigned at submission.
        seq: u64,
        /// Request that owns the sequence.
        request_id: RequestId,
        /// Sampled token, already appended to the sequence's KV blocks.
        token: Token,
    },
    /// The sequence finished, either by reaching its cap or by sampling only stops.
    Completed {
        /// Stable scheduler sequence id assigned at submission.
        seq: u64,
        /// Request that owns the sequence.
        request_id: RequestId,
        /// All tokens emitted for this request, in order.
        tokens: Vec<Token>,
    },
    /// There are no waiting or running requests.
    Idle {
        /// Current waiting-queue depth.
        queued: usize,
    },
    /// Requests are waiting but none can advance with the current KV budget.
    Blocked {
        /// Current waiting-queue depth.
        queued: usize,
        /// Number of running sequences.
        running: usize,
    },
}

#[derive(Debug)]
struct SequenceSlot {
    seq: u64,
    kv_seq: Option<u64>,
    request: GenerationRequest,
    sampler: Sampler,
    context: Vec<i32>,
    generated: Vec<Token>,
    limit: usize,
}

/// A bounded, deterministic continuous-batching scheduler.
#[derive(Debug)]
pub struct SequenceScheduler {
    config: SchedulerConfig,
    kv: BlockTableKvCache,
    waiting: VecDeque<SequenceSlot>,
    running: VecDeque<SequenceSlot>,
    next_seq: u64,
}

impl SequenceScheduler {
    /// Creates a scheduler with the bounds in `config`.
    ///
    /// # Errors
    ///
    /// Propagates [`EngineError::InvalidBlockSize`] and
    /// [`EngineError::CacheInvariant`] from the underlying KV cache.
    pub fn new(config: SchedulerConfig) -> Result<Self, EngineError> {
        let kv = BlockTableKvCache::new(config.block_size, config.kv_blocks)?;
        Ok(Self {
            config,
            kv,
            waiting: VecDeque::new(),
            running: VecDeque::new(),
            next_seq: 1,
        })
    }

    /// Enqueues a generation request and returns its stable scheduler id.
    ///
    /// The request is admitted to the running set by the next `step` when both
    /// the running limit and KV budget allow it.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::QueueFull`] when the bounded queue is full,
    /// [`EngineError::InvalidParams`] for rejected sampling parameters and
    /// [`EngineError::CacheInvariant`] if the sequence id space is exhausted.
    pub fn submit(&mut self, request: GenerationRequest) -> Result<u64, EngineError> {
        if self.waiting.len() >= self.config.max_queue_depth {
            return Err(EngineError::QueueFull {
                depth: self.waiting.len(),
                capacity: self.config.max_queue_depth,
            });
        }
        request
            .params
            .validate()
            .map_err(|_| EngineError::InvalidParams {
                reason: "quarrel-core rejected the sampling parameters",
            })?;
        let param_limit = to_usize(request.params.max_tokens)?;
        let limit = request.max_tokens.min(param_limit);

        let seq = self.next_seq;
        self.next_seq = seq.checked_add(1).ok_or(EngineError::CacheInvariant {
            reason: "sequence id space exhausted",
        })?;
        self.waiting.push_back(SequenceSlot {
            seq,
            kv_seq: None,
            sampler: Sampler::new(request.seed),
            context: request.prompt.clone(),
            generated: Vec::new(),
            limit,
            request,
        });
        Ok(seq)
    }

    /// Advances the next running sequence by one `backend` forward pass.
    ///
    /// Logits are audited and sampled with the request's own sampler and
    /// parameters. A sampler response containing only stop tokens completes the
    /// sequence without emitting a token.
    ///
    /// # Errors
    ///
    /// Propagates backend, logit, sampler and KV errors.
    pub fn step(&mut self, backend: &dyn Backend) -> Result<StepOutcome, EngineError> {
        self.advance(|slot| {
            let logits = backend.forward(&slot.context)?;
            if let HealthStatus::Unhealthy(reason) = audit_logits(&logits) {
                return Err(EngineError::BackendLogits {
                    backend: backend.name().to_owned(),
                    reason,
                });
            }
            match slot
                .sampler
                .sample(&logits, &slot.request.params, &slot.context)
            {
                Ok(token) => Ok(Some(token)),
                Err(EngineError::StopTokenSampled { .. }) => Ok(None),
                Err(error) => Err(error),
            }
        })
    }

    /// Advances the next running sequence using a caller-supplied token source.
    ///
    /// The callback receives the scheduler sequence id, the current context
    /// (prompt plus generated ids) and the number of tokens still allowed.
    /// Returning `Ok(None)` completes the sequence without emitting a token.
    /// This is useful for deterministic tests and for backends that do not use
    /// [`crate::sampler::Sampler`].
    ///
    /// # Errors
    ///
    /// Propagates callback errors and KV errors.
    pub fn step_with<F>(&mut self, mut produce: F) -> Result<StepOutcome, EngineError>
    where
        F: FnMut(u64, &[i32], usize) -> Result<Option<Token>, EngineError>,
    {
        self.advance(|slot| {
            let remaining = slot.limit.saturating_sub(slot.generated.len());
            produce(slot.seq, slot.context.as_slice(), remaining)
        })
    }

    /// Returns the current waiting-queue depth, including preempted requests.
    #[must_use]
    pub fn queue_depth(&self) -> usize {
        self.waiting.len()
    }

    /// Returns the configured waiting-queue capacity.
    #[must_use]
    pub fn queue_capacity(&self) -> usize {
        self.config.max_queue_depth
    }

    /// Returns the number of currently running sequences.
    #[must_use]
    pub fn running_len(&self) -> usize {
        self.running.len()
    }

    /// Returns `true` when neither waiting nor running sequences remain.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.waiting.is_empty() && self.running.is_empty()
    }

    /// Returns a read-only view of the scheduler's KV cache.
    #[must_use]
    pub fn kv_cache(&self) -> &BlockTableKvCache {
        &self.kv
    }

    /// Copies the tokens generated so far by `seq`.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::UnknownSequence`] if the id is unknown or its
    /// request has completed and been removed.
    pub fn generated(&self, seq: u64) -> Result<Vec<Token>, EngineError> {
        self.find_slot(seq)
            .map(|slot| slot.generated.clone())
            .ok_or(EngineError::UnknownSequence { seq })
    }

    /// Cancels `seq` and returns the tokens it generated before cancellation.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::UnknownSequence`] if the id is unknown or its
    /// request has already completed, and propagates KV release errors.
    pub fn cancel(&mut self, seq: u64) -> Result<Vec<Token>, EngineError> {
        if let Some(position) = self.running.iter().position(|slot| slot.seq == seq) {
            if let Some(kv_seq) = self.running.get(position).and_then(|slot| slot.kv_seq) {
                self.kv.free(kv_seq)?;
            }
            let slot = self
                .running
                .remove(position)
                .ok_or(EngineError::CacheInvariant {
                    reason: "running position disappeared during cancellation",
                })?;
            return Ok(slot.generated);
        }
        if let Some(position) = self.waiting.iter().position(|slot| slot.seq == seq) {
            let slot = self
                .waiting
                .remove(position)
                .ok_or(EngineError::CacheInvariant {
                    reason: "waiting position disappeared during cancellation",
                })?;
            return Ok(slot.generated);
        }
        Err(EngineError::UnknownSequence { seq })
    }

    fn advance<F>(&mut self, mut produce: F) -> Result<StepOutcome, EngineError>
    where
        F: FnMut(&mut SequenceSlot) -> Result<Option<Token>, EngineError>,
    {
        self.admit_waiting()?;
        if self.running.is_empty() {
            return Ok(if self.waiting.is_empty() {
                StepOutcome::Idle {
                    queued: self.waiting.len(),
                }
            } else {
                self.blocked_outcome()
            });
        }

        let front = self.running.front().ok_or(EngineError::CacheInvariant {
            reason: "running set disappeared before stepping",
        })?;
        if front.generated.len() >= front.limit {
            let tokens = front.generated.clone();
            return self.complete_front(tokens);
        }
        let seq = front.seq;
        let extra_blocks = self.blocks_needed_for_next_token(seq)?;
        if extra_blocks > self.kv.free_blocks() && !self.preempt_one(seq)? {
            return Ok(self.blocked_outcome());
        }

        let sampled = {
            let slot = self
                .running
                .front_mut()
                .ok_or(EngineError::CacheInvariant {
                    reason: "running set disappeared before producing a token",
                })?;
            produce(slot)?
        };
        let Some(token) = sampled else {
            return self.complete_front(Vec::new());
        };

        let (seq, kv_seq) = {
            let slot = self.running.front().ok_or(EngineError::CacheInvariant {
                reason: "running set disappeared after producing a token",
            })?;
            let kv_seq = slot.kv_seq.ok_or(EngineError::CacheInvariant {
                reason: "running sequence has no KV allocation",
            })?;
            (slot.seq, kv_seq)
        };
        self.kv.append(kv_seq, &[token.id])?;
        {
            let slot = self
                .running
                .front_mut()
                .ok_or(EngineError::CacheInvariant {
                    reason: "running set disappeared after appending a token",
                })?;
            slot.context.push(token.id);
            slot.generated.push(token.clone());
        }

        let completed = {
            let slot = self.running.front().ok_or(EngineError::CacheInvariant {
                reason: "running set disappeared after updating a token",
            })?;
            slot.generated.len() >= slot.limit
        };
        if completed {
            let tokens = self
                .running
                .front()
                .ok_or(EngineError::CacheInvariant {
                    reason: "running set disappeared before completion",
                })?
                .generated
                .clone();
            return self.complete_front(tokens);
        }

        self.running.rotate_left(1);
        Ok(StepOutcome::Progress {
            seq,
            request_id: self
                .request_id_of(seq)
                .ok_or(EngineError::UnknownSequence { seq })?,
            token,
        })
    }

    fn admit_waiting(&mut self) -> Result<(), EngineError> {
        while self.running.len() < self.config.max_running {
            let required = match self.waiting.front() {
                Some(slot) => slot.context.len().div_ceil(self.config.block_size),
                None => break,
            };
            if required > self.kv.free_blocks() {
                break;
            }
            let mut slot = match self.waiting.pop_front() {
                Some(slot) => slot,
                None => break,
            };
            let kv_seq = match self.kv.allocate(&slot.request.id, slot.context.len()) {
                Ok(kv_seq) => kv_seq,
                Err(EngineError::CapacityExceeded { .. }) => {
                    self.waiting.push_front(slot);
                    break;
                }
                Err(error) => return Err(error),
            };
            if let Err(error) = self.kv.append(kv_seq, &slot.context) {
                let _ = self.kv.free(kv_seq);
                return Err(error);
            }
            slot.kv_seq = Some(kv_seq);
            self.running.push_back(slot);
        }
        Ok(())
    }

    fn preempt_one(&mut self, protected_seq: u64) -> Result<bool, EngineError> {
        if self.waiting.len() >= self.config.max_queue_depth {
            return Ok(false);
        }
        let index = match self
            .running
            .iter()
            .rposition(|slot| slot.seq != protected_seq && slot.kv_seq.is_some())
        {
            Some(index) => index,
            None => return Ok(false),
        };
        let kv_seq = match self.running.get(index).and_then(|slot| slot.kv_seq) {
            Some(kv_seq) => kv_seq,
            None => return Ok(false),
        };
        self.kv.free(kv_seq)?;
        if let Some(slot) = self.running.get_mut(index) {
            slot.kv_seq = None;
        }
        let slot = self
            .running
            .remove(index)
            .ok_or(EngineError::CacheInvariant {
                reason: "preemption position disappeared",
            })?;
        self.waiting.push_back(slot);
        Ok(true)
    }

    fn complete_front(&mut self, tokens: Vec<Token>) -> Result<StepOutcome, EngineError> {
        let front = self.running.front().ok_or(EngineError::CacheInvariant {
            reason: "running set disappeared before completion",
        })?;
        let seq = front.seq;
        if let Some(kv_seq) = front.kv_seq {
            self.kv.free(kv_seq)?;
        }
        let slot = self
            .running
            .pop_front()
            .ok_or(EngineError::CacheInvariant {
                reason: "running set disappeared while completing",
            })?;
        Ok(StepOutcome::Completed {
            seq,
            request_id: slot.request.id,
            tokens,
        })
    }

    fn blocks_needed_for_next_token(&self, seq: u64) -> Result<usize, EngineError> {
        let (kv_seq, length) = self
            .running
            .iter()
            .find(|slot| slot.seq == seq)
            .map(|slot| (slot.kv_seq, slot.context.len()))
            .ok_or(EngineError::UnknownSequence { seq })?;
        let kv_seq = kv_seq.ok_or(EngineError::CacheInvariant {
            reason: "running sequence has no KV allocation",
        })?;
        let next_length = length.checked_add(1).ok_or(EngineError::CacheInvariant {
            reason: "sequence length overflow",
        })?;
        let owned_blocks = self.kv.block_table(kv_seq)?.len();
        Ok(next_length
            .div_ceil(self.config.block_size)
            .saturating_sub(owned_blocks))
    }

    fn find_slot(&self, seq: u64) -> Option<&SequenceSlot> {
        self.running
            .iter()
            .chain(self.waiting.iter())
            .find(|slot| slot.seq == seq)
    }

    fn request_id_of(&self, seq: u64) -> Option<RequestId> {
        self.find_slot(seq).map(|slot| slot.request.id.clone())
    }

    fn blocked_outcome(&self) -> StepOutcome {
        StepOutcome::Blocked {
            queued: self.waiting.len(),
            running: self.running.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{GenerationRequest, SchedulerConfig, SequenceScheduler, StepOutcome};
    use crate::EngineError;
    use crate::backend::ReferenceBackend;
    use quarrel_core::{RequestId, SamplingParams, Token};
    use std::collections::HashMap;

    fn request(id: &str, prompt: &[i32], max_tokens: usize) -> GenerationRequest {
        let mut params = SamplingParams::new();
        params.temperature = 0.0;
        params.stop_token_ids.clear();
        GenerationRequest::new(
            RequestId::new(id).expect("non-empty id"),
            prompt.to_vec(),
            max_tokens,
            params,
            7,
        )
    }

    fn context_token(
        _seq: u64,
        context: &[i32],
        _remaining: usize,
    ) -> Result<Option<Token>, EngineError> {
        Ok(Some(Token::new(100 + context.len() as i32, 0.0)))
    }

    #[test]
    fn round_robin_makes_fair_progress_and_completes_all_requests() {
        let mut scheduler =
            SequenceScheduler::new(SchedulerConfig::new(8, 3, 4, 64)).expect("scheduler");
        let first = scheduler.submit(request("a", &[1], 3)).expect("submit a");
        let second = scheduler.submit(request("b", &[2], 3)).expect("submit b");
        let third = scheduler.submit(request("c", &[3], 3)).expect("submit c");
        assert_eq!(scheduler.queue_depth(), 3);

        let mut seen = Vec::new();
        let mut tokens_per_seq: HashMap<u64, i32> = HashMap::new();
        let mut completions = 0;
        for _ in 0..9 {
            match scheduler.step_with(context_token).expect("step") {
                StepOutcome::Progress { seq, token, .. } => {
                    seen.push(seq);
                    let produced = tokens_per_seq.entry(seq).or_insert(0);
                    *produced += 1;
                    assert_eq!(token.id, 100 + *produced);
                }
                StepOutcome::Completed { seq, tokens, .. } => {
                    seen.push(seq);
                    completions += 1;
                    assert_eq!(tokens.len(), 3);
                }
                other => panic!("unexpected outcome: {other:?}"),
            }
        }
        assert_eq!(
            seen,
            vec![
                first, second, third, first, second, third, first, second, third
            ]
        );
        assert_eq!(completions, 3);
        assert!(scheduler.is_empty());
        assert_eq!(scheduler.kv_cache().blocks_used(), 0);
    }

    #[test]
    fn preemption_preserves_state_and_resumes_the_victim() {
        let mut scheduler =
            SequenceScheduler::new(SchedulerConfig::new(4, 2, 2, 4)).expect("scheduler");
        let first = scheduler
            .submit(request("a", &[1, 2], 4))
            .expect("submit a");
        let victim = scheduler
            .submit(request("b", &[3, 4], 4))
            .expect("submit b");

        scheduler.step_with(context_token).expect("step a");
        scheduler.step_with(context_token).expect("step b");
        scheduler.step_with(context_token).expect("step a again");
        scheduler.step_with(context_token).expect("step b again");
        assert_eq!(scheduler.kv_cache().blocks_used(), 4);

        let outcome = scheduler.step_with(context_token).expect("preempting step");
        assert!(matches!(outcome, StepOutcome::Progress { seq, .. } if seq == first));
        assert_eq!(scheduler.running_len(), 1);
        assert_eq!(scheduler.queue_depth(), 1);
        let victim_tokens = scheduler.generated(victim).expect("victim state");
        assert_eq!(victim_tokens.len(), 2);
        assert_eq!(victim_tokens[0].id, 102);
        assert_eq!(victim_tokens[1].id, 103);
        assert_eq!(scheduler.kv_cache().blocks_used(), 3);

        for _ in 0..8 {
            let _ = scheduler.step_with(context_token).expect("later step");
        }
        assert!(scheduler.is_empty());
        assert_eq!(scheduler.kv_cache().blocks_used(), 0);
    }

    #[test]
    fn bounded_queue_rejects_overflow_with_a_typed_error() {
        let mut scheduler =
            SequenceScheduler::new(SchedulerConfig::new(1, 1, 4, 16)).expect("scheduler");
        scheduler
            .submit(request("first", &[1], 2))
            .expect("first request");
        assert_eq!(scheduler.queue_depth(), 1);
        assert_eq!(scheduler.queue_capacity(), 1);

        assert!(matches!(
            scheduler.submit(request("second", &[2], 2)),
            Err(EngineError::QueueFull {
                depth: 1,
                capacity: 1
            })
        ));
        scheduler.step_with(context_token).expect("admit and step");
        assert_eq!(scheduler.queue_depth(), 0);
        scheduler
            .submit(request("second", &[2], 2))
            .expect("slot freed by admission");
        assert_eq!(scheduler.running_len(), 1);
    }

    #[test]
    fn completion_releases_kv_and_idle_is_reported() {
        let mut scheduler =
            SequenceScheduler::new(SchedulerConfig::new(2, 1, 4, 8)).expect("scheduler");
        let seq = scheduler.submit(request("done", &[5], 2)).expect("submit");
        let first = scheduler.step_with(context_token).expect("first token");
        assert!(matches!(
            first,
            StepOutcome::Progress { seq: progress, .. } if progress == seq
        ));
        let last = scheduler.step_with(context_token).expect("last token");
        match last {
            StepOutcome::Completed {
                seq: completed,
                tokens,
                ..
            } => {
                assert_eq!(completed, seq);
                assert_eq!(tokens.len(), 2);
            }
            other => panic!("unexpected outcome: {other:?}"),
        }
        assert!(scheduler.is_empty());
        assert_eq!(scheduler.kv_cache().blocks_used(), 0);
        assert!(matches!(
            scheduler.step_with(context_token).expect("idle step"),
            StepOutcome::Idle { queued: 0 }
        ));
    }

    #[test]
    fn unknown_sequence_is_a_typed_error() {
        let mut scheduler =
            SequenceScheduler::new(SchedulerConfig::new(2, 1, 4, 8)).expect("scheduler");
        assert!(matches!(
            scheduler.cancel(99),
            Err(EngineError::UnknownSequence { seq: 99 })
        ));
        assert!(matches!(
            scheduler.generated(99),
            Err(EngineError::UnknownSequence { seq: 99 })
        ));
    }

    #[test]
    fn step_uses_a_real_backend_and_sampler() {
        let mut scheduler =
            SequenceScheduler::new(SchedulerConfig::new(2, 1, 8, 8)).expect("scheduler");
        scheduler
            .submit(request("backend", &[1, 2, 3], 2))
            .expect("submit");
        let backend = ReferenceBackend::new();
        let first = scheduler.step(&backend).expect("first backend step");
        let second = scheduler.step(&backend).expect("second backend step");
        assert!(matches!(first, StepOutcome::Progress { .. }));
        assert!(matches!(second, StepOutcome::Completed { .. }));
        assert!(scheduler.is_empty());
    }

    #[test]
    fn stop_only_sampling_completes_without_tokens() {
        let mut request = request("stops", &[1], 3);
        request.params.stop_token_ids = (0..32).collect();
        let mut scheduler =
            SequenceScheduler::new(SchedulerConfig::new(2, 1, 4, 8)).expect("scheduler");
        scheduler.submit(request).expect("submit");
        assert!(matches!(
            scheduler.step(&ReferenceBackend::new()).expect("step"),
            StepOutcome::Completed { tokens, .. } if tokens.is_empty()
        ));
    }
}
