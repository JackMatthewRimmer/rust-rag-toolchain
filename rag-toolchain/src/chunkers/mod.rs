/// # Chunkers
/// Module to contain all the methods of chunking allowing for
/// prepping text before embedding and storing it.
mod token_chunker;

pub use token_chunker::{ChunkingError, TokenChunker};
