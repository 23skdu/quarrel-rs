#![no_main]

use libfuzzer_sys::fuzz_target;

use quarrel_flight::{
    embed_request_schema, generate_request_schema, ipc_to_batch, kv_blocks_schema,
};

// Arrow IPC bytes arrive from other Agents over Flight. Decoding arbitrary bytes must stay inside
// the typed FlightError path; the helper catches panics from the Arrow decoder as a backstop.
fuzz_target!(|data: &[u8]| {
    let schema = match data.first() {
        Some(1) => embed_request_schema(),
        Some(2) => kv_blocks_schema(),
        _ => generate_request_schema(),
    };
    let _ = ipc_to_batch(data, schema);
});
