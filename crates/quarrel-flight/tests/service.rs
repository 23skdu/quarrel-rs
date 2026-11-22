//! Direct trait tests plus an in-process tonic FlightClient round trip.

use std::sync::Arc;

use arrow::array::UInt64Array;
use arrow_flight::{
    Action, Criteria, FlightClient, FlightDescriptor, Ticket, encode::FlightDataEncoderBuilder,
    flight_service_server::FlightService as _,
};
use bytes::Bytes;
use futures::{StreamExt, TryStreamExt};
use quarrel_core::{GenerateRequest, ModelId, RequestId, SamplingParams};
use quarrel_engine::backend::{Backend, ReferenceBackend, generate};
use quarrel_flight::{
    DEFAULT_QUEUE_DEPTH, GENERATE_PATH, HEALTH_ACTION, QuarrelFlightService, decode_finished,
    decode_generate_request, decode_generate_response, encode_generate_request, flight_server,
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
