mod character_chunker;
mod token_chunker;
/// # Chunkers
/// Module to contain all the methods of chunking allowing for
/// prepping text before embedding and storing it.
mod traits;

pub use token_chunker::{ChunkingError, TokenChunker};
