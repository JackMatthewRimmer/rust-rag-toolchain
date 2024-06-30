use crate::common::{Chunk, Chunks};
use futures::Stream;
use std::error::Error;

pub trait Chunker {
    type ErrorType: Error;
    fn generate_chunks(&self, raw_text: &str) -> Result<Chunks, Self::ErrorType>;
}

#[allow(unused)]
pub trait StreamedChunker {
    type ErrorType: Error;
    type CharacterStream: Stream<Item = std::io::Result<char>>;
    type ChunkStream: Stream<Item = Result<Chunk, Self::ErrorType>>;
    fn generate_chunks(&self, data_stream: Self::CharacterStream) -> Self::ChunkStream;
}
