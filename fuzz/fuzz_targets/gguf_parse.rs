#![no_main]

use libfuzzer_sys::fuzz_target;

use quarrel_gguf::GgufFile;

// GGUF metadata is untrusted: the parser must reject arbitrary bytes with a typed error and never
// panic, allocate an unbounded buffer, or index outside the input.
fuzz_target!(|data: &[u8]| {
    if let Ok(file) = GgufFile::from_bytes(data) {
        // Touch the parsed descriptors so fuzzing also covers metadata lookups and tensor views.
        for descriptor in &file.header().tensors {
            let _ = descriptor.dtype.block_bytes();
            let _ = file.tensor(&descriptor.name);
        }
    }
});
