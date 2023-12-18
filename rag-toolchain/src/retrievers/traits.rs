use crate::common::types::Chunk;
use async_trait::async_trait;
use std::error::Error;

/// # EmbeddingRetriever
/// Trait for stores that allow for similar text to be retrieved using embeddings
#[async_trait]
pub trait AsyncRetriever {
    type ErrorType: Error;
    async fn retrieve(&self, text: &str) -> Result<Chunk, Self::ErrorType>;
}
