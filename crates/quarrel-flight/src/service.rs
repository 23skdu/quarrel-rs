//! In-memory Arrow Flight service for generation requests.

use std::{
    collections::HashMap,
    pin::Pin,
    sync::{Arc, Mutex},
};

use arrow::record_batch::RecordBatch;
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
    FlightError, decode_generate_request, encode_generate_response, generate_response_schema,
    schema_hash,
};

/// Descriptor path used for generation requests.
pub const GENERATE_PATH: &str = "/quarrel/v1/generate";

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
}

struct Inner {
    requests: Mutex<HashMap<String, RequestState>>,
    backend: Arc<dyn Backend + Send + Sync>,
    queue_depth: usize,
}

/// An Arrow Flight service with in-memory, request-keyed response queues.
///
/// `DoPut` accepts a generate-request batch, runs the configured backend and
/// stores the resulting token batches. `DoGet` streams those batches back to
/// the agent using the request id as the ticket.
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
        let schema = generate_response_schema();
        let queued = self.queued_batches(request_id);
        let mut info = FlightInfo::new()
            .try_with_schema(schema.as_ref())
            .expect("response schema is encodable")
            .with_descriptor(FlightDescriptor::new_path(vec![
                GENERATE_PATH.to_owned(),
                request_id.to_owned(),
            ]))
            .with_endpoint(FlightEndpoint::new().with_ticket(Ticket::new(request_id.to_owned())))
            .with_total_records(i64::try_from(queued).unwrap_or(-1));
        info.ordered = true;
        info
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
        if last == GENERATE_PATH {
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
        let batches = self
            .lock_requests()
            .get(&request_id)
            .map(|state| state.batches.clone())
            .ok_or_else(|| Status::not_found(format!("unknown request: {request_id}")))?;

        let input = stream::iter(batches.into_iter().map(Ok));
        let encoded = FlightDataEncoderBuilder::new()
            .with_schema(generate_response_schema())
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
        if !descriptor.path.is_empty() && descriptor.path[0] != GENERATE_PATH {
            return Err(Status::unimplemented(format!(
                "unsupported descriptor: {}",
                descriptor.path.join("/")
            )));
        }

        let batches = utils::flight_data_to_batches(&frames)
            .map_err(|err| FlightError::Transport(err.to_string()))?;
        let mut enqueued = Vec::new();
        for batch in batches {
            let decoded = decode_generate_request(&batch).map_err(FlightError::to_status)?;
            let request_id = decoded.request_id.as_str().to_owned();
            let params = decoded.params.clone();
            let tokens: Vec<Token> = generate(
                self.inner.backend.as_ref(),
                &decoded.token_ids,
                &params,
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
            let responses = tokens
                .into_iter()
                .enumerate()
                .map(|(index, token)| {
                    encode_generate_response(
                        &decoded.request_id,
                        u64::try_from(index + 1).unwrap_or(u64::MAX),
                        &token,
                        index + 1 == total,
                    )
                    .map_err(FlightError::to_status)
                })
                .collect::<Result<Vec<_>, Status>>()?;
            self.lock_requests()
                .insert(request_id.clone(), RequestState { batches: responses });
            enqueued.push(request_id);
        }

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
