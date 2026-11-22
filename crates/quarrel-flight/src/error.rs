//! Typed errors for the Flight plane.

/// Errors produced by the Arrow Flight schemas, adapters and service.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum FlightError {
    /// An Arrow operation failed.
    #[error("arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),

    /// A Flight transport operation failed.
    #[error("flight transport error: {0}")]
    Transport(String),

    /// An IPC payload could not be decoded (or panicked during decoding).
    #[error("invalid IPC payload: {0}")]
    InvalidPayload(String),

    /// A required column or field was missing from a batch.
    #[error("missing field: {0}")]
    MissingField(&'static str),

    /// A batch did not have the schema the adapter expects.
    #[error("schema mismatch: expected {expected}, got {actual}")]
    SchemaMismatch {
        /// Canonical schema hash of the expected schema.
        expected: String,
        /// Canonical schema hash of the actual schema.
        actual: String,
    },

    /// A batch had an unexpected number of rows.
    #[error("expected a single row, got {0}")]
    ExpectedSingleRow(usize),

    /// A Flight descriptor path or ticket was not recognised.
    #[error("unknown flight descriptor: {0}")]
    UnknownDescriptor(String),

    /// A generation request is not active (or has already been consumed).
    #[error("unknown request: {0}")]
    UnknownRequest(String),

    /// An action or descriptor argument was malformed.
    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    /// The engine rejected a forward pass.
    #[error("engine error: {0}")]
    Engine(String),

    /// A KV-cache dtype string was not one of the canonical Flight dtypes.
    #[error("unsupported KV cache dtype: {0}")]
    UnsupportedKvDtype(String),

    /// A KV-cache dtype requires a host byte order this target does not have.
    #[error("KV cache dtype {dtype} requires {expected}-endian host, running on {actual}-endian")]
    EndiannessMismatch {
        /// Canonical dtype string from the batch.
        dtype: &'static str,
        /// Byte order required by the payload encoding.
        expected: &'static str,
        /// Byte order of the host interpreting the payload.
        actual: &'static str,
    },
}

impl FlightError {
    /// Convert into a gRPC status, preserving a useful code where possible.
    pub fn to_status(self) -> tonic::Status {
        let message = self.to_string();
        match self {
            Self::UnknownDescriptor(_) | Self::UnknownRequest(_) => {
                tonic::Status::not_found(message)
            }
            Self::InvalidArgument(_)
            | Self::ExpectedSingleRow(_)
            | Self::MissingField(_)
            | Self::UnsupportedKvDtype(_) => tonic::Status::invalid_argument(message),
            Self::SchemaMismatch { .. } | Self::EndiannessMismatch { .. } => {
                tonic::Status::failed_precondition(message)
            }
            _ => tonic::Status::internal(message),
        }
    }
}

impl From<FlightError> for tonic::Status {
    fn from(err: FlightError) -> Self {
        err.to_status()
    }
}

impl From<arrow_flight::error::FlightError> for FlightError {
    fn from(err: arrow_flight::error::FlightError) -> Self {
        Self::Transport(err.to_string())
    }
}

impl From<tonic::Status> for FlightError {
    fn from(status: tonic::Status) -> Self {
        Self::Transport(status.to_string())
    }
}
