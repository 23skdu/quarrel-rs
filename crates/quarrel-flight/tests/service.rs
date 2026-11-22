//! Direct trait tests plus an in-process tonic FlightClient round trip.

use std::sync::Arc;

use arrow::array::{Int32Array, StringArray, UInt64Array};
use arrow::record_batch::RecordBatch;
use arrow_flight::{
    Action, Criteria, FlightClient, FlightData, FlightDescriptor, FlightInfo, Ticket,
    encode::FlightDataEncoderBuilder, flight_service_server::FlightService as _,
};
use bytes::Bytes;
use futures::{StreamExt, TryStreamExt};
use quarrel_core::{GenerateRequest, ModelId, RequestId, SamplingParams};
use quarrel_engine::backend::{Backend, ReferenceBackend, generate};
use quarrel_flight::{
    DEFAULT_QUEUE_DEPTH, EMBED_PATH, GENERATE_PATH, HEALTH_ACTION, QuarrelFlightService,
    decode_embedding, decode_finished, decode_generate_request, decode_generate_response,
    embed_request_schema, embed_response_schema, encode_embed_request, encode_generate_request,
    flight_server, generate_response_schema, schema_hash,
};
use tonic::Request;

fn request(id: &str) -> GenerateRequest {
    GenerateRequest::new(
        RequestId::new(id).unwrap(),
        ModelId::new("reference").unwrap(),
        vec![128_000, 271, 42],
        SamplingParams::new(),
        true,
    )
    .unwrap()
}

fn deterministic_seed(request_id: &str) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in request_id.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

#[tokio::test]
async fn health_action_and_flight_descriptors() {
    let service = QuarrelFlightService::default();

    let response = service
        .do_action(Request::new(Action {
            r#type: HEALTH_ACTION.to_owned(),
            body: Bytes::new(),
        }))
        .await
        .unwrap();
    let result = response.into_inner().next().await.unwrap().unwrap();
    let body = String::from_utf8(result.body.to_vec()).unwrap();
    assert!(body.contains("\"ready\":true"));
    assert!(body.contains("\"schema_hash\""));
    assert!(body.contains("generate_request"));

    let flights = service
        .list_flights(Request::new(Criteria::default()))
        .await
        .unwrap();
    assert_eq!(flights.into_inner().count().await, 0);

    let unknown = service
        .get_flight_info(Request::new(FlightDescriptor::new_path(vec![
            GENERATE_PATH.to_owned(),
            "missing".to_owned(),
        ])))
        .await;
    assert!(unknown.is_err());

    let actions = service
        .list_actions(Request::new(arrow_flight::Empty {}))
        .await
        .unwrap();
    assert_eq!(actions.into_inner().count().await, 1);
}

#[tokio::test]
async fn in_process_tonic_do_put_do_get_round_trip() {
    let service = QuarrelFlightService::with_backend(Arc::new(ReferenceBackend::new()));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();

    let server = tokio::spawn(async move {
        let incoming = async_stream::stream! {
            loop {
                match listener.accept().await {
                    Ok((socket, _)) => yield Ok::<_, std::io::Error>(socket),
                    Err(err) => {
                        yield Err(err);
                        break;
                    }
                }
            }
        };
        tonic::transport::Server::builder()
            .add_service(flight_server(service))
            .serve_with_incoming(incoming)
            .await
            .unwrap();
    });

    let channel = tonic::transport::Endpoint::from_shared(format!("http://{address}"))
        .unwrap()
        .connect()
        .await
        .unwrap();
    let mut client = FlightClient::new(channel);

    let generated = request("agent-grpc-1");
    let batch = encode_generate_request(&generated).unwrap();
    let uploads = FlightDataEncoderBuilder::new()
        .with_flight_descriptor(Some(FlightDescriptor::new_path(vec![
            GENERATE_PATH.to_owned(),
        ])))
        .build(futures::stream::iter(vec![Ok::<
            _,
            arrow_flight::error::FlightError,
        >(batch)]));
    let mut put_results = client.do_put(uploads).await.unwrap();
    let put = put_results.next().await.unwrap().unwrap();
    assert_eq!(
        String::from_utf8(put.app_metadata.to_vec()).unwrap(),
        "agent-grpc-1"
    );

    let stream = client.do_get(Ticket::new("agent-grpc-1")).await.unwrap();
    let batches: Vec<_> = stream.try_collect().await.unwrap();
    let expected: Vec<_> = generate(
        &ReferenceBackend::new() as &dyn Backend,
        &generated.token_ids,
        &generated.params,
        deterministic_seed("agent-grpc-1"),
        DEFAULT_QUEUE_DEPTH,
    )
    .unwrap();
    assert_eq!(batches.len(), expected.len());

    let mut tokens = Vec::new();
    let mut finishes = Vec::new();
    for (index, batch) in batches.iter().enumerate() {
        tokens.extend(decode_generate_response(batch).unwrap());
        finishes.extend(decode_finished(batch).unwrap());
        let sequences = batch
            .column(1)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        assert_eq!(sequences.value(0), u64::try_from(index + 1).unwrap());
    }
    assert_eq!(tokens, expected);
    assert_eq!(finishes.len(), expected.len());
    assert!(finishes.last().copied().unwrap_or(false));
    assert!(
        finishes[..finishes.len().saturating_sub(1)]
            .iter()
            .all(|finished| !finished)
    );

    let mut health = client
        .do_action(Action {
            r#type: HEALTH_ACTION.to_owned(),
            body: Bytes::new(),
        })
        .await
        .unwrap();
    let health_body = health.next().await.unwrap().unwrap();
    assert!(
        String::from_utf8(health_body.to_vec())
            .unwrap()
            .contains("\"ready\":true")
    );

    server.abort();
}

#[tokio::test]
async fn service_rejects_unknown_ticket() {
    let service = QuarrelFlightService::default();
    let result = service
        .do_get(Request::new(Ticket::new("no-such-request")))
        .await;
    assert!(result.is_err());
}

/// Expected server embedding: first four backend logits, L2-normalised in f64.
fn expected_embedding(token_id: i32) -> Vec<f32> {
    let logits: Vec<f32> = ReferenceBackend::new()
        .forward(&[token_id])
        .unwrap()
        .into_iter()
        .take(4)
        .collect();
    let norm = logits
        .iter()
        .fold(0.0f64, |sum, value| {
            sum + f64::from(*value) * f64::from(*value)
        })
        .sqrt();
    logits
        .iter()
        .map(|value| (f64::from(*value) / norm) as f32)
        .collect()
}

fn uploads(
    batch: RecordBatch,
    path: &str,
) -> impl futures::Stream<Item = Result<FlightData, arrow_flight::error::FlightError>> + Send {
    let descriptor = FlightDescriptor::new_path(vec![path.to_owned()]);
    FlightDataEncoderBuilder::new()
        .with_flight_descriptor(Some(descriptor))
        .build(futures::stream::iter(vec![Ok(batch)]))
}

async fn spawn_server(service: QuarrelFlightService) -> (String, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let incoming = async_stream::stream! {
            loop {
                match listener.accept().await {
                    Ok((socket, _)) => yield Ok::<_, std::io::Error>(socket),
                    Err(err) => {
                        yield Err(err);
                        break;
                    }
                }
            }
        };
        tonic::transport::Server::builder()
            .add_service(flight_server(service))
            .serve_with_incoming(incoming)
            .await
            .unwrap();
    });
    (address.to_string(), server)
}

#[tokio::test]
async fn embed_do_put_do_get_round_trip() {
    let service = QuarrelFlightService::default();
    let state = service.clone();
    let (address, server) = spawn_server(service).await;
    let channel = tonic::transport::Endpoint::from_shared(format!("http://{address}"))
        .unwrap()
        .connect()
        .await
        .unwrap();
    let mut client = FlightClient::new(channel);

    let id = "agent-embed-1";
    let tokens = [11, -7, 42];
    let batch = RecordBatch::try_new(
        embed_request_schema(),
        vec![
            Arc::new(StringArray::from(vec![id; tokens.len()])),
            Arc::new(Int32Array::from(tokens.to_vec())),
        ],
    )
    .unwrap();
    let mut put_results = client.do_put(uploads(batch, EMBED_PATH)).await.unwrap();
    let result = put_results.next().await.unwrap().unwrap();
    assert_eq!(String::from_utf8(result.app_metadata.to_vec()).unwrap(), id);
    assert_eq!(state.active_requests(), vec![id.to_owned()]);
    assert_eq!(state.queued_batches(id), tokens.len());

    let descriptor = FlightDescriptor::new_path(vec![EMBED_PATH.to_owned(), id.to_owned()]);
    let info = state
        .get_flight_info(Request::new(descriptor.clone()))
        .await
        .unwrap()
        .into_inner();
    let expected_schema = FlightInfo::new()
        .try_with_schema(embed_response_schema().as_ref())
        .unwrap()
        .schema;
    assert_eq!(info.schema, expected_schema);
    assert_eq!(info.flight_descriptor, Some(descriptor));
    assert_eq!(info.total_records, i64::try_from(tokens.len()).unwrap());

    let stream = client.do_get(Ticket::new(id)).await.unwrap();
    let batches: Vec<_> = stream.try_collect().await.unwrap();
    assert_eq!(batches.len(), tokens.len());
    for (batch, token) in batches.iter().zip(tokens) {
        assert_eq!(batch.schema(), embed_response_schema());
        let (request_id, token_id, values) = decode_embedding(batch).unwrap();
        assert_eq!(request_id, id);
        assert_eq!(token_id, token);
        assert_eq!(values.len(), 4);
        for (actual, expected) in values.iter().zip(expected_embedding(token)) {
            assert!((actual - expected).abs() < 1e-6, "{actual} vs {expected}");
        }
        let norm: f64 = values
            .iter()
            .fold(0.0, |sum, value| {
                sum + f64::from(*value) * f64::from(*value)
            })
            .sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    // The generate path still uses its own response schema.
    let generate_info = state
        .get_flight_info(Request::new(FlightDescriptor::new_path(vec![
            GENERATE_PATH.to_owned(),
            "missing-generate".to_owned(),
        ])))
        .await;
    assert!(generate_info.is_err());
    assert_ne!(
        schema_hash(&embed_response_schema()),
        schema_hash(&generate_response_schema())
    );
    server.abort();
}

#[tokio::test]
async fn embed_round_trip_over_tonic_flight_client() {
    let service = QuarrelFlightService::default();
    let (address, server) = spawn_server(service).await;
    let channel = tonic::transport::Endpoint::from_shared(format!("http://{address}"))
        .unwrap()
        .connect()
        .await
        .unwrap();
    let mut client = FlightClient::new(channel);

    let id = "agent-embed-grpc";
    let batch = encode_embed_request(&RequestId::new(id).unwrap(), 271).unwrap();
    let mut put_results = client.do_put(uploads(batch, EMBED_PATH)).await.unwrap();
    let put = put_results.next().await.unwrap().unwrap();
    assert_eq!(String::from_utf8(put.app_metadata.to_vec()).unwrap(), id);

    let stream = client.do_get(Ticket::new(id)).await.unwrap();
    let batches: Vec<_> = stream.try_collect().await.unwrap();
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].schema(), embed_response_schema());
    let (request_id, token_id, values) = decode_embedding(&batches[0]).unwrap();
    assert_eq!(request_id, id);
    assert_eq!(token_id, 271);
    let expected = expected_embedding(271);
    for (actual, expected) in values.iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-6);
    }
    server.abort();
}

fn assert_tonic_code(error: arrow_flight::error::FlightError, expected: tonic::Code) {
    match error {
        arrow_flight::error::FlightError::Tonic(status) => assert_eq!(status.code(), expected),
        other => panic!("unexpected Flight client error: {other}"),
    }
}

#[tokio::test]
async fn embed_error_paths_return_typed_statuses() {
    let service = QuarrelFlightService::default();
    let (address, server) = spawn_server(service.clone()).await;
    let channel = tonic::transport::Endpoint::from_shared(format!("http://{address}"))
        .unwrap()
        .connect()
        .await
        .unwrap();
    let mut client = FlightClient::new(channel);

    let id = RequestId::new("agent-embed-bad").unwrap();
    match client
        .do_put(uploads(
            encode_embed_request(&id, 1).unwrap(),
            "/quarrel/v1/unknown",
        ))
        .await
    {
        Err(error) => assert_tonic_code(error, tonic::Code::Unimplemented),
        Ok(_) => panic!("unsupported descriptor was accepted"),
    }

    // A generate batch sent to the embed path fails the exact-schema check.
    match client
        .do_put(uploads(
            encode_generate_request(&request("agent-embed-wrong-schema")).unwrap(),
            EMBED_PATH,
        ))
        .await
    {
        Err(error) => assert_tonic_code(error, tonic::Code::FailedPrecondition),
        Ok(_) => panic!("generate batch was accepted on the embed path"),
    }

    let empty_id = RecordBatch::try_new(
        embed_request_schema(),
        vec![
            Arc::new(StringArray::from(vec![""])),
            Arc::new(Int32Array::from(vec![1])),
        ],
    )
    .unwrap();
    match client.do_put(uploads(empty_id, EMBED_PATH)).await {
        Err(error) => assert_tonic_code(error, tonic::Code::InvalidArgument),
        Ok(_) => panic!("empty request id was accepted"),
    }

    assert!(
        service
            .do_get(Request::new(Ticket::new("no-such-embed")))
            .await
            .is_err()
    );
    assert!(
        service
            .get_flight_info(Request::new(FlightDescriptor::new_path(vec![
                EMBED_PATH.to_owned(),
                "no-such-embed".to_owned(),
            ])))
            .await
            .is_err()
    );
    server.abort();
}

#[test]
fn generated_batches_are_keyed_by_request_id() {
    // Direct adapter/service state check without a gRPC transport.
    let service = QuarrelFlightService::default();
    let generated = request("agent-direct-1");
    let batch = encode_generate_request(&generated).unwrap();
    let decoded = decode_generate_request(&batch).unwrap();
    assert_eq!(decoded.request_id.as_str(), "agent-direct-1");
    assert!(service.active_requests().is_empty());
}
