use crate::common::Chunks;
use std::error::Error;
use std::io::Read;

pub trait Chunker {
    type ErrorType: Error;
    fn generate_chunks(&self, raw_text: &str) -> Result<Chunks, Self::ErrorType>;
}

pub trait StreamedChunker {
    type ErrorType: Error;
    fn generate_chunks(&self, data_stream: impl Read) -> Result<Chunks, Self::ErrorType>;
}
