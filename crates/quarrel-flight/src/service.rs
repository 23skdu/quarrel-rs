//! In-memory Arrow Flight service for generation requests.

use std::{
    collections::HashMap,
    pin::Pin,
    sync::{Arc, Mutex},
};

use arrow::{datatypes::SchemaRef, record_batch::RecordBatch};
use arrow_flight::{
    Action, ActionType, Criteria, Empty, FlightData, FlightDescriptor, FlightEndpoint, FlightInfo,
    HandshakeResponse, PollInfo, PutResult, SchemaResult, Ticket, encode::FlightDataEncoderBuilder,
    flight_service_server::FlightService, utils,
};
use bytes::Bytes;
use futures::{Stream, StreamExt, stream};
use quarrel_core::Token;
use quarrel_engine::backend::{Backend, ReferenceBackend, generate};
use tonic::{Request, Response, Status};

use crate::{
    DEFAULT_EMBEDDING_DIM, FlightError, GenerateResponseBuilder, decode_embed_request_rows,
    decode_generate_request_view, embedding_response_schema_for, encode_embedding,
    generate_response_schema, record_batch_stream, schema_hash,
};

/// Descriptor path used for generation requests.
pub const GENERATE_PATH: &str = "/quarrel/v1/generate";

/// Descriptor path used for embedding requests.
pub const EMBED_PATH: &str = "/quarrel/v1/embed";

/// Health action path.
pub const HEALTH_ACTION: &str = "/quarrel/v1/health";

/// Maximum number of response batches retained per request.
pub const DEFAULT_QUEUE_DEPTH: usize = 64;

type FlightDataStream = Pin<Box<dyn Stream<Item = Result<FlightData, Status>> + Send>>;
type PutResultStream = Pin<Box<dyn Stream<Item = Result<PutResult, Status>> + Send>>;
type ActionResultStream = Pin<Box<dyn Stream<Item = Result<arrow_flight::Result, Status>> + Send>>;
type FlightInfoStream = Pin<Box<dyn Stream<Item = Result<FlightInfo, Status>> + Send>>;
type HandshakeStream = Pin<Box<dyn Stream<Item = Result<HandshakeResponse, Status>> + Send>>;

#[derive(Debug, Clone)]
struct RequestState {
    batches: Vec<RecordBatch>,
    /// Response schema stored alongside the batches so `DoGet` and
    /// `GetFlightInfo` describe generate and embed responses correctly.
    schema: SchemaRef,
    /// Descriptor path of the request that produced the batches.
    path: &'static str,
}

struct Inner {
    requests: Mutex<HashMap<String, RequestState>>,
    backend: Arc<dyn Backend + Send + Sync>,
    queue_depth: usize,
}

/// An Arrow Flight service with in-memory, request-keyed response queues.
///
/// `DoPut` accepts generate-request batches at [`GENERATE_PATH`] and
/// embed-request batches at [`EMBED_PATH`]. Both run the configured backend and
/// store the resulting response batches with their own Arrow schema. `DoGet`
/// streams those batches back to the agent using the request id as the ticket.
#[derive(Clone)]
pub struct QuarrelFlightService {
    inner: Arc<Inner>,
}

impl Default for QuarrelFlightService {
    fn default() -> Self {
        Self::with_backend(Arc::new(ReferenceBackend::new()))
    }
}

impl QuarrelFlightService {
    /// Build a service using a specific backend.
    #[must_use]
    pub fn with_backend(backend: Arc<dyn Backend + Send + Sync>) -> Self {
        Self {
            inner: Arc::new(Inner {
                requests: Mutex::new(HashMap::new()),
                backend,
                queue_depth: DEFAULT_QUEUE_DEPTH,
            }),
        }
    }

    /// Ids of requests with at least one queued response batch.
    #[must_use]
    pub fn active_requests(&self) -> Vec<String> {
        self.lock_requests().keys().cloned().collect::<Vec<_>>()
    }

    /// Number of response batches currently queued for a request.
    #[must_use]
    pub fn queued_batches(&self, request_id: &str) -> usize {
        self.lock_requests()
            .get(request_id)
            .map_or(0, |state| state.batches.len())
    }

    /// Remove all queued responses for a request.
    pub fn remove_request(&self, request_id: &str) {
        self.lock_requests().remove(request_id);
    }

    fn lock_requests(&self) -> std::sync::MutexGuard<'_, HashMap<String, RequestState>> {
        self.inner
            .requests
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn health_json(&self) -> String {
        let schemas = [
            (
                "generate_request",
                schema_hash(&crate::generate_request_schema()),
            ),
            (
                "generate_response",
                schema_hash(&generate_response_schema()),
            ),
            ("embed_request", schema_hash(&crate::embed_request_schema())),
            (
                "embed_response",
                schema_hash(&crate::embed_response_schema()),
            ),
            ("kv_blocks", schema_hash(&crate::kv_blocks_schema())),
        ];
        let mut out = String::from("{\"ready\":true,\"backend\":\"");
        push_json_string(&mut out, self.inner.backend.name());
        out.push_str("\",\"model\":\"reference\",\"schema_hash\":\"");
        push_json_string(&mut out, &schema_hash(&generate_response_schema()));
        out.push_str("\",\"schemas\":{");
        for (i, (name, hash)) in schemas.iter().enumerate() {
            if i != 0 {
                out.push(',');
            }
            out.push('"');
            push_json_string(&mut out, name);
            out.push_str("\":\"");
            push_json_string(&mut out, hash);
            out.push('"');
        }
        out.push_str("}}");
        out
    }

    fn describe(&self, request_id: &str) -> FlightInfo {
        let state = self
            .lock_requests()
            .get(request_id)
            .cloned()
            .unwrap_or_else(|| RequestState {
                batches: Vec::new(),
                schema: generate_response_schema(),
                path: GENERATE_PATH,
            });
        self.describe_state(request_id, &state)
    }

    fn describe_state(&self, request_id: &str, state: &RequestState) -> FlightInfo {
        let queued = state.batches.len();
        let mut info = FlightInfo::new()
            .try_with_schema(state.schema.as_ref())
            .expect("response schema is encodable")
            .with_descriptor(FlightDescriptor::new_path(vec![
                state.path.to_owned(),
                request_id.to_owned(),
            ]))
            .with_endpoint(FlightEndpoint::new().with_ticket(Ticket::new(request_id.to_owned())))
            .with_total_records(i64::try_from(queued).unwrap_or(-1));
        info.ordered = true;
        info
    }

    /// Derive the canonical 4-dimensional embedding for one token.
    ///
    /// The first `DEFAULT_EMBEDDING_DIM` logits from the backend are copied into
    /// a vector and L2-normalised. A non-finite or zero norm falls back to a
    /// deterministic one-hot vector selected by the token id.
    fn embed_token(&self, token_id: i32) -> Result<Vec<f32>, FlightError> {
        let logits = self
            .inner
            .backend
            .forward(&[token_id])
            .map_err(|err| FlightError::Engine(err.to_string()))?;
        let dim = usize::try_from(DEFAULT_EMBEDDING_DIM).unwrap_or(4);
        let mut vector = vec![0.0f32; dim];
        for (slot, value) in vector.iter_mut().zip(logits.iter()) {
            *slot = *value;
        }
        let norm_squared = vector.iter().fold(0.0f64, |sum, value| {
            sum + f64::from(*value) * f64::from(*value)
        });
        let norm = norm_squared.sqrt();
        if !norm.is_finite() || norm <= 0.0 {
            let index = token_id
                .rem_euclid(DEFAULT_EMBEDDING_DIM)
                .clamp(0, DEFAULT_EMBEDDING_DIM - 1) as usize;
            vector.fill(0.0);
            if let Some(slot) = vector.get_mut(index) {
                *slot = 1.0;
            }
            return Ok(vector);
        }
        for value in &mut vector {
            *value = (f64::from(*value) / norm) as f32;
        }
        Ok(vector)
    }

    fn enqueue_generate(&self, batches: Vec<RecordBatch>) -> Result<Vec<String>, Status> {
        let mut enqueued = Vec::new();
        for batch in batches {
            let decoded = decode_generate_request_view(&batch).map_err(FlightError::to_status)?;
            let request_id = decoded.request_id.as_str().to_owned();
            // The prompt is borrowed directly from the Arrow buffer; no token
            // vector is copied on the request path.
            let tokens: Vec<Token> = generate(
                self.inner.backend.as_ref(),
                decoded.token_ids,
                &decoded.params,
                deterministic_seed(&request_id),
                self.inner.queue_depth,
            )
            .map_err(|err| FlightError::Engine(err.to_string()))?
            .into_iter()
            .take(self.inner.queue_depth)
            .collect();

            if tokens.is_empty() {
                return Err(Status::invalid_argument(
                    "backend produced no tokens for request",
                ));
            }
            let total = tokens.len();
            let mut builder = GenerateResponseBuilder::new().map_err(FlightError::to_status)?;
            let responses = tokens
                .into_iter()
                .enumerate()
                .map(|(index, token)| {
                    builder
                        .encode(
                            &decoded.request_id,
                            u64::try_from(index + 1).unwrap_or(u64::MAX),
                            &token,
                            index + 1 == total,
                        )
                        .map_err(FlightError::to_status)
                })
                .collect::<Result<Vec<_>, Status>>()?;
            self.lock_requests().insert(
                request_id.clone(),
                RequestState {
                    batches: responses,
                    schema: generate_response_schema(),
                    path: GENERATE_PATH,
                },
            );
            enqueued.push(request_id);
        }
        Ok(enqueued)
    }

    fn enqueue_embed(&self, batches: Vec<RecordBatch>) -> Result<Vec<String>, Status> {
        let mut enqueued = Vec::new();
        for batch in batches {
            let rows = decode_embed_request_rows(&batch).map_err(FlightError::to_status)?;
            if rows.is_empty() {
                return Err(Status::invalid_argument("embed request has no rows"));
            }
            // Group by request id while preserving first-seen order so the
            // DoPut result metadata is deterministic.
            let mut grouped: Vec<(String, Vec<RecordBatch>)> = Vec::new();
            for (request_id, token_id) in rows {
                let embedding = self.embed_token(token_id).map_err(FlightError::to_status)?;
                let response = encode_embedding(&request_id, token_id, &embedding)
                    .map_err(FlightError::to_status)?;
                let id = request_id.as_str().to_owned();
                match grouped.iter_mut().find(|(existing, _)| existing == &id) {
                    Some((_, responses)) => responses.push(response),
                    None => grouped.push((id, vec![response])),
                }
            }
            let schema =
                embedding_response_schema_for(usize::try_from(DEFAULT_EMBEDDING_DIM).unwrap_or(4));
            for (request_id, responses) in grouped {
                self.lock_requests().insert(
                    request_id.clone(),
                    RequestState {
                        batches: responses,
                        schema: schema.clone(),
                        path: EMBED_PATH,
                    },
                );
                enqueued.push(request_id);
            }
        }
        Ok(enqueued)
    }
}

fn push_json_string(out: &mut String, value: &str) {
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if c.is_control() => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
}

fn request_id_from_descriptor(descriptor: &FlightDescriptor) -> Option<String> {
    if !descriptor.path.is_empty() {
        let last = descriptor.path.last()?;
        if last == GENERATE_PATH || last == EMBED_PATH {
            return None;
        }
        return Some(last.clone());
    }
    let cmd = String::from_utf8(descriptor.cmd.to_vec()).ok()?;
    if cmd.is_empty() { None } else { Some(cmd) }
}

fn deterministic_seed(request_id: &str) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in request_id.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

#[tonic::async_trait]
impl FlightService for QuarrelFlightService {
    type HandshakeStream = HandshakeStream;
    type ListFlightsStream = FlightInfoStream;
    type DoGetStream = FlightDataStream;
    type DoPutStream = PutResultStream;
    type DoExchangeStream = FlightDataStream;
    type DoActionStream = ActionResultStream;
    type ListActionsStream = Pin<Box<dyn Stream<Item = Result<ActionType, Status>> + Send>>;

    async fn handshake(
        &self,
        _request: Request<tonic::Streaming<arrow_flight::HandshakeRequest>>,
    ) -> Result<Response<Self::HandshakeStream>, Status> {
        Err(Status::unimplemented("handshake is not required"))
    }

    async fn list_flights(
        &self,
        request: Request<Criteria>,
    ) -> Result<Response<Self::ListFlightsStream>, Status> {
        let criteria = String::from_utf8_lossy(&request.into_inner().expression).into_owned();
        let ids = self.active_requests();
        let infos: Vec<Result<FlightInfo, Status>> = ids
            .into_iter()
            .filter(|id| criteria.is_empty() || id.contains(&criteria))
            .map(|id| Ok(self.describe(&id)))
            .collect();
        Ok(Response::new(Box::pin(stream::iter(infos))))
    }

    async fn get_flight_info(
        &self,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        let descriptor = request.into_inner();
        let id = request_id_from_descriptor(&descriptor)
            .ok_or_else(|| Status::invalid_argument("missing request id in descriptor"))?;
        if self.queued_batches(&id) == 0 && !self.active_requests().contains(&id) {
            return Err(Status::not_found(format!("unknown request: {id}")));
        }
        Ok(Response::new(self.describe(&id)))
    }

    async fn poll_flight_info(
        &self,
        _request: Request<FlightDescriptor>,
    ) -> Result<Response<PollInfo>, Status> {
        Err(Status::unimplemented("poll_flight_info is not required"))
    }

    async fn get_schema(
        &self,
        _request: Request<FlightDescriptor>,
    ) -> Result<Response<SchemaResult>, Status> {
        Err(Status::unimplemented("use GetFlightInfo instead"))
    }

    async fn do_get(
        &self,
        request: Request<Ticket>,
    ) -> Result<Response<Self::DoGetStream>, Status> {
        let request_id = String::from_utf8(request.into_inner().ticket.to_vec())
            .map_err(|_| Status::invalid_argument("ticket must be a UTF-8 request id"))?;
        let state = self
            .lock_requests()
            .get(&request_id)
            .cloned()
            .ok_or_else(|| Status::not_found(format!("unknown request: {request_id}")))?;

        // Move the batch vector into a RecordBatchStream; Arrow payload buffers
        // stay shared and are not copied before the Flight encoder sees them.
        let input = record_batch_stream(state.batches).map(Ok);
        let encoded = FlightDataEncoderBuilder::new()
            .with_schema(state.schema)
            .build(input)
            .map(|result| result.map_err(Status::from));
        Ok(Response::new(Box::pin(encoded)))
    }

    async fn do_put(
        &self,
        request: Request<tonic::Streaming<FlightData>>,
    ) -> Result<Response<Self::DoPutStream>, Status> {
        let mut inbound = request.into_inner();
        let mut frames = Vec::new();
        while let Some(frame) = inbound.message().await? {
            frames.push(frame);
        }
        let descriptor = frames
            .first()
            .and_then(|frame| frame.flight_descriptor.clone())
            .ok_or_else(|| Status::invalid_argument("DoPut stream has no descriptor"))?;
        let path = descriptor.path.first().map(String::as_str);
        let enqueued = match path {
            None | Some(GENERATE_PATH) => {
                let batches = utils::flight_data_to_batches(&frames)
                    .map_err(|err| FlightError::Transport(err.to_string()))?;
                self.enqueue_generate(batches)?
            }
            Some(EMBED_PATH) => {
                let batches = utils::flight_data_to_batches(&frames)
                    .map_err(|err| FlightError::Transport(err.to_string()))?;
                self.enqueue_embed(batches)?
            }
            Some(other) => {
                return Err(Status::unimplemented(format!(
                    "unsupported descriptor: {other}"
                )));
            }
        };

        let metadata = enqueued.join(",");
        let results = async_stream::stream! {
            yield Ok::<PutResult, Status>(PutResult {
                app_metadata: Bytes::from(metadata),
            });
        };
        Ok(Response::new(Box::pin(results)))
    }

    async fn do_exchange(
        &self,
        _request: Request<tonic::Streaming<FlightData>>,
    ) -> Result<Response<Self::DoExchangeStream>, Status> {
        Err(Status::unimplemented("do_exchange is not required"))
    }

    async fn do_action(
        &self,
        request: Request<Action>,
    ) -> Result<Response<Self::DoActionStream>, Status> {
        let action = request.into_inner();
        if action.r#type != HEALTH_ACTION {
            return Err(Status::invalid_argument(format!(
                "unknown action: {}",
                action.r#type
            )));
        }
        let body = Bytes::from(self.health_json());
        Ok(Response::new(Box::pin(stream::once(async move {
            Ok(arrow_flight::Result { body })
        }))))
    }

    async fn list_actions(
        &self,
        _request: Request<Empty>,
    ) -> Result<Response<Self::ListActionsStream>, Status> {
        let action = ActionType {
            r#type: HEALTH_ACTION.to_owned(),
            description: "Readiness and Arrow schema hashes".to_owned(),
        };
        Ok(Response::new(Box::pin(stream::iter(vec![Ok(action)]))))
    }
}

impl std::fmt::Debug for QuarrelFlightService {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("QuarrelFlightService")
            .field("backend", &self.inner.backend.name())
            .field("requests", &self.lock_requests().len())
            .finish()
    }
}

impl std::fmt::Display for QuarrelFlightService {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "QuarrelFlightService(backend={}, requests={})",
            self.inner.backend.name(),
            self.lock_requests().len()
        )
    }
}

/// Build a tonic server for the Flight service.
#[must_use]
pub fn flight_server(
    service: QuarrelFlightService,
) -> arrow_flight::flight_service_server::FlightServiceServer<QuarrelFlightService> {
    arrow_flight::flight_service_server::FlightServiceServer::new(service)
}
