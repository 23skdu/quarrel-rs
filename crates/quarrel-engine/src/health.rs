//! Lightweight logit auditing.
//!
//! This is not a statistical test; it catches the failure modes that make
//! sampling meaningless: non-finite values, degenerate distributions and
//! absurd magnitudes.

/// Result of [`audit_logits`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HealthStatus {
    /// The logits look usable.
    Healthy,
    /// The logits are usable but suspicious, with an explanation.
    Suspicious(String),
    /// The logits must not be sampled from, with an explanation.
    Unhealthy(String),
}

/// Audits a logit slice for values that would corrupt sampling.
///
/// Non-finite values and magnitudes above `1e6` are unhealthy. An empty slice
/// or a slice whose entries are all identical is suspicious but not fatal:
/// uniform sampling from equal logits is well-defined.
#[must_use]
pub fn audit_logits(logits: &[f32]) -> HealthStatus {
    for (index, &value) in logits.iter().enumerate() {
        if !value.is_finite() {
            return HealthStatus::Unhealthy(format!("logit at index {index} is not finite"));
        }
        if value.abs() > 1e6 {
            return HealthStatus::Unhealthy(format!(
                "logit at index {index} has extreme magnitude {value}"
            ));
        }
    }
    if logits.is_empty() {
        return HealthStatus::Suspicious("logits are empty".to_owned());
    }
    let first = logits[0];
    if logits.iter().all(|&value| value == first) {
        return HealthStatus::Suspicious("all logits are identical".to_owned());
    }
    HealthStatus::Healthy
}

#[cfg(test)]
mod tests {
    use super::{HealthStatus, audit_logits};

    #[test]
    fn healthy_logits() {
        assert_eq!(audit_logits(&[-2.0, 0.0, 3.0]), HealthStatus::Healthy);
        assert_eq!(audit_logits(&[0.0, 1e6]), HealthStatus::Healthy);
    }

    #[test]
    fn suspicious_logits() {
        assert_eq!(
            audit_logits(&[2.0, 2.0, 2.0]),
            HealthStatus::Suspicious("all logits are identical".to_owned())
        );
        assert_eq!(
            audit_logits(&[]),
            HealthStatus::Suspicious("logits are empty".to_owned())
        );
    }

    #[test]
    fn unhealthy_logits() {
        assert!(matches!(
            audit_logits(&[f32::NAN]),
            HealthStatus::Unhealthy(_)
        ));
        assert!(matches!(
            audit_logits(&[f32::NEG_INFINITY]),
            HealthStatus::Unhealthy(_)
        ));
        assert!(matches!(
            audit_logits(&[0.0, 1_000_001.0]),
            HealthStatus::Unhealthy(_)
        ));
        assert!(matches!(
            audit_logits(&[-1_000_001.0]),
            HealthStatus::Unhealthy(_)
        ));
    }
}
