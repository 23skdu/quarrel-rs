//! Memory-mapped files and borrowed tensor views.

use crate::dtype::DType;
use crate::error::GgufError;
use crate::header::{GgufHeader, TensorDescriptor, parse_header};
use memmap2::{Mmap, MmapOptions};
use std::ops::Range;
use std::path::Path;

/// A parsed GGUF header plus a borrowed view of the underlying bytes.
pub struct GgufFile<'a> {
    bytes: &'a [u8],
    header: GgufHeader,
}

impl<'a> GgufFile<'a> {
    /// Parse a complete GGUF file from memory.
    pub fn from_bytes(bytes: &'a [u8]) -> Result<Self, GgufError> {
        let header = parse_header(bytes)?;
        tracing::trace!(
            version = header.version.number(),
            tensors = header.tensor_count,
            data_offset = header.data_offset,
            "parsed GGUF header"
        );
        Ok(Self { bytes, header })
    }

    /// Parsed header information.
    pub fn header(&self) -> &GgufHeader {
        &self.header
    }

    /// Borrow one tensor's data, with its descriptor.
    pub fn tensor(&self, name: &str) -> Result<TensorView<'_>, GgufError> {
        let descriptor = self.tensor_descriptor(name)?;
        let range = self.tensor_range(descriptor)?;
        Ok(TensorView {
            descriptor,
            bytes: &self.bytes[range],
        })
    }

    /// Borrow only the bytes of the named tensor.
    pub fn tensor_bytes(&self, name: &str) -> Result<&'a [u8], GgufError> {
        let descriptor = self.tensor_descriptor(name)?;
        let range = self.tensor_range(descriptor)?;
        Ok(&self.bytes[range])
    }

    fn tensor_descriptor(&self, name: &str) -> Result<&TensorDescriptor, GgufError> {
        self.header
            .tensors
            .iter()
            .find(|tensor| tensor.name == name)
            .ok_or_else(|| GgufError::TensorNotFound(name.to_owned()))
    }

    fn tensor_range(&self, descriptor: &TensorDescriptor) -> Result<Range<usize>, GgufError> {
        let file_len = self.bytes.len() as u64;
        let start = self
            .header
            .data_offset
            .checked_add(descriptor.offset)
            .ok_or(GgufError::Overflow)?;
        let end = start
            .checked_add(descriptor.byte_len)
            .ok_or(GgufError::Overflow)?;
        if end > file_len {
            return Err(GgufError::BadTensorOffset {
                name: descriptor.name.clone(),
                data_offset: self.header.data_offset,
                offset: descriptor.offset,
                byte_len: descriptor.byte_len,
                file_len,
            });
        }
        let start = usize::try_from(start).map_err(|_| GgufError::Overflow)?;
        let end = usize::try_from(end).map_err(|_| GgufError::Overflow)?;
        Ok(start..end)
    }
}

/// An owned, read-only memory map backed by a GGUF file.
///
/// The file is mapped directly into the process address space; tensor views borrow slices of that
/// mapping and never copy weight bytes into the heap. Model files are therefore an immutable input
/// to the process: do not modify or truncate them while a `GgufMmap` is alive.
pub struct GgufMmap {
    mmap: Mmap,
    header: GgufHeader,
}

impl GgufMmap {
    /// Open and parse a GGUF file.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, GgufError> {
        let path = path.as_ref();
        let mmap = map_file(path)?;
        let header = parse_header(&mmap)?;
        tracing::trace!(
            path = %path.display(),
            tensors = header.tensor_count,
            "memory-mapped GGUF file"
        );
        Ok(Self { mmap, header })
    }

    /// Get a borrowed view of the mapped bytes and header.
    ///
    /// The header is cloned so that the returned view can own its parsed
    /// metadata while the byte slice remains tied to this map.
    pub fn file(&self) -> GgufFile<'_> {
        GgufFile {
            bytes: &self.mmap[..],
            header: self.header.clone(),
        }
    }

    /// Borrow one tensor's data, with its descriptor, without cloning metadata.
    pub fn tensor(&self, name: &str) -> Result<TensorView<'_>, GgufError> {
        let descriptor = find_tensor(&self.header, name)?;
        let range = mapped_tensor_range(&self.header, self.mmap.len(), descriptor)?;
        Ok(TensorView {
            descriptor,
            bytes: &self.mmap[range],
        })
    }

    /// Borrow only the bytes of the named tensor without cloning metadata.
    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8], GgufError> {
        let descriptor = find_tensor(&self.header, name)?;
        let range = mapped_tensor_range(&self.header, self.mmap.len(), descriptor)?;
        Ok(&self.mmap[range])
    }

    /// Parsed header information.
    pub fn header(&self) -> &GgufHeader {
        &self.header
    }

    /// Number of tensor descriptors advertised by the file header.
    pub fn metadata_tensor_count(&self) -> u64 {
        self.header.tensor_count
    }
}

fn find_tensor<'a>(header: &'a GgufHeader, name: &str) -> Result<&'a TensorDescriptor, GgufError> {
    header
        .tensors
        .iter()
        .find(|tensor| tensor.name == name)
        .ok_or_else(|| GgufError::TensorNotFound(name.to_owned()))
}

fn mapped_tensor_range(
    header: &GgufHeader,
    file_len: usize,
    descriptor: &TensorDescriptor,
) -> Result<Range<usize>, GgufError> {
    let file_len = file_len as u64;
    let start = header
        .data_offset
        .checked_add(descriptor.offset)
        .ok_or(GgufError::Overflow)?;
    let end = start
        .checked_add(descriptor.byte_len)
        .ok_or(GgufError::Overflow)?;
    if end > file_len {
        return Err(GgufError::BadTensorOffset {
            name: descriptor.name.clone(),
            data_offset: header.data_offset,
            offset: descriptor.offset,
            byte_len: descriptor.byte_len,
            file_len,
        });
    }
    let start = usize::try_from(start).map_err(|_| GgufError::Overflow)?;
    let end = usize::try_from(end).map_err(|_| GgufError::Overflow)?;
    Ok(start..end)
}

/// Map a GGUF file into memory without copying its payload.
fn map_file(path: impl AsRef<Path>) -> Result<Mmap, GgufError> {
    let file = std::fs::File::open(path)?;
    let len = file.metadata()?.len();
    if len == 0 {
        return Err(GgufError::Truncated {
            offset: 0,
            needed: 4,
        });
    }
    // Validate that the file length fits the address space before mapping it.
    let _ = usize::try_from(len).map_err(|_| GgufError::Overflow)?;

    // SAFETY: the file is opened read-only and the model registry treats GGUF files as immutable
    // inputs. No code in this crate writes through the mapping, and the mapping is dropped before a
    // model file can be replaced or removed. This is the OS `mmap` FFI shim for zero-copy weights.
    let mmap = unsafe { MmapOptions::new().map(&file) }?;
    Ok(mmap)
}

/// A zero-copy view of one tensor.
pub struct TensorView<'a> {
    descriptor: &'a TensorDescriptor,
    bytes: &'a [u8],
}

impl<'a> TensorView<'a> {
    /// Tensor name.
    pub fn name(&self) -> &str {
        &self.descriptor.name
    }

    /// Tensor dimensions.
    pub fn shape(&self) -> &[u64] {
        &self.descriptor.shape
    }

    /// Tensor element type.
    pub fn dtype(&self) -> DType {
        self.descriptor.dtype
    }

    /// Raw tensor bytes.
    pub fn as_bytes(&self) -> &'a [u8] {
        self.bytes
    }

    /// Number of logical elements implied by the shape.
    pub fn element_count(&self) -> u64 {
        self.descriptor
            .shape
            .iter()
            .fold(1u64, |acc, &dim| acc.saturating_mul(dim))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture::fixture_bytes;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_file(bytes: &[u8]) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("quarrel-file-{nanos}-{}.gguf", std::process::id()));
        std::fs::write(&path, bytes).expect("write fixture");
        path
    }

    #[test]
    fn borrowed_file_exposes_tensor_views_and_raw_bytes() {
        let bytes = fixture_bytes(crate::GgufVersion::V2);
        let file = GgufFile::from_bytes(&bytes).expect("parse fixture");
        assert_eq!(file.header().tensor_count, 2);

        let view = file.tensor("token_embeddings.weight").expect("tensor view");
        assert_eq!(view.name(), "token_embeddings.weight");
        assert_eq!(view.shape(), &[4, 8]);
        assert_eq!(view.dtype(), DType::Q8_0);
        assert_eq!(view.element_count(), 32);
        assert_eq!(view.as_bytes().len(), 34);

        let raw = file
            .tensor_bytes("output.weight")
            .expect("raw tensor bytes");
        assert_eq!(raw.len(), 18);
        assert_eq!(raw[0], 0x00);
        assert_eq!(raw[1], 0x40);

        assert!(file.tensor("missing").is_err());
        assert!(file.tensor_bytes("missing").is_err());
    }

    #[test]
    fn mmap_open_exposes_header_and_zero_copy_views() {
        let bytes = fixture_bytes(crate::GgufVersion::V3);
        let path = temp_file(&bytes);
        let mapped = GgufMmap::open(&path).expect("memory-map fixture");
        assert_eq!(mapped.header().tensor_count, 2);
        assert_eq!(mapped.metadata_tensor_count(), 2);

        let view = mapped.tensor("output.weight").expect("mapped tensor");
        assert_eq!(view.name(), "output.weight");
        assert_eq!(view.shape(), &[2, 8]);
        assert_eq!(view.dtype(), DType::Q4_0);
        assert_eq!(view.element_count(), 16);
        assert_eq!(
            view.as_bytes(),
            mapped.tensor_bytes("output.weight").unwrap()
        );

        let borrowed = mapped.file();
        assert_eq!(
            borrowed.tensor("token_embeddings.weight").unwrap().shape(),
            &[4, 8]
        );
        assert_eq!(borrowed.header().version, crate::GgufVersion::V3);

        assert!(mapped.tensor("missing").is_err());
        assert!(mapped.tensor_bytes("missing").is_err());
        drop(mapped);
        std::fs::remove_file(path).expect("remove fixture");
    }

    #[test]
    fn open_reports_empty_and_missing_files() {
        let empty = temp_file(&[]);
        assert!(GgufMmap::open(&empty).is_err());
        std::fs::remove_file(empty).expect("remove empty fixture");

        let missing = std::env::temp_dir().join("quarrel-does-not-exist.gguf");
        assert!(GgufMmap::open(missing).is_err());
    }

    #[test]
    fn truncated_payload_reports_bad_tensor_offset() {
        let bytes = fixture_bytes(crate::GgufVersion::V2);
        let parsed = GgufFile::from_bytes(&bytes).expect("parse fixture");
        let header = parsed.header().clone();
        let truncated = &bytes[..bytes.len() - 20];
        // Keep the already-parsed header but expose a short payload so the
        // borrowed view's bounds check is exercised directly.
        let file = GgufFile {
            bytes: truncated,
            header,
        };
        assert!(file.tensor("token_embeddings.weight").is_err());
        assert!(file.tensor_bytes("output.weight").is_err());
    }
}
