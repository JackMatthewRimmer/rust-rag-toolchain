mod character_chunker;
mod token_chunker;
mod traits;
pub use character_chunker::CharacterChunker;
/// # Chunkers
/// Module to contain all the methods of chunking allowing for
/// prepping text before embedding and storing it.
pub use token_chunker::{TokenChunker, TokenChunkingError};
pub use traits::Chunker;
