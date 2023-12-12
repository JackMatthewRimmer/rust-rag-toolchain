use crate::common::types::{Chunk, Chunks, Embedding};
use async_trait::async_trait;
use std::error::Error;

/// # AsyncEmbeddingClient
/// Trait for any client that generates embeddings asynchronously
/// such as OpenAI
#[async_trait]
pub trait AsyncEmbeddingClient {
    type ErrorType: Error;
    async fn generate_embedding(&self, text: Chunk) -> Result<(Chunk, Embedding), Self::ErrorType>;
    async fn generate_embeddings(
        &self,
        text: Chunks,
    ) -> Result<Vec<(Chunk, Embedding)>, Self::ErrorType>;
}

/// # EmbeddingClient
/// Trait for any client that generates embeddings synchronously
/// can be used for any local embedding model
pub trait EmbeddingClient {
    type ErrorType: Error;
    fn generate_embedding(&self, text: Chunk) -> Result<(Chunk, Embedding), Self::ErrorType>;
    fn generate_embeddings(&self, text: Chunks)
        -> Result<Vec<(Chunk, Embedding)>, Self::ErrorType>;
}
