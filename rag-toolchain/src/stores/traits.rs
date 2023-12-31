use crate::common::types::{Chunk, Embedding};
use async_trait::async_trait;
use std::error::Error;

/// # EmbeddingStore
/// Trait for struct that allows embeddings to be wrote to an
/// external persistent store. Mainly a vector database.
#[async_trait]
pub trait EmbeddingStore {
    // The custom error type for the store
    type ErrorType: Error;
    /// Takes an embedding and writes it to an external source
    async fn store(&self, embedding: (Chunk, Embedding)) -> Result<(), Self::ErrorType>;
    async fn store_batch(&self, embeddings: Vec<(Chunk, Embedding)>)
        -> Result<(), Self::ErrorType>;
}
