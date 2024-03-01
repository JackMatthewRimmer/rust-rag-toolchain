use crate::common::{Chunk, Embedding};
use std::error::Error;
use std::future::Future;

/// # EmbeddingStore
/// Trait for struct that allows embeddings to be wrote to an
/// external persistent store. Mainly a vector database.
pub trait EmbeddingStore {
    // The custom error type for the store
    type ErrorType: Error;
    /// Takes an embedding and writes it to an external source
    fn store(
        &self,
        embedding: (Chunk, Embedding),
    ) -> impl Future<Output = Result<(), Self::ErrorType>> + Send;
    fn store_batch(
        &self,
        embeddings: Vec<(Chunk, Embedding)>,
    ) -> impl Future<Output = Result<(), Self::ErrorType>> + Send;
}

#[cfg(test)]
use mockall::*;
#[cfg(test)]
mock! {
    pub EmbeddingStore {}
    impl EmbeddingStore for EmbeddingStore {
        type ErrorType = std::io::Error;
        async fn store(&self, embedding: (Chunk, Embedding)) -> Result<(), <Self as EmbeddingStore>::ErrorType>;
        async fn store_batch(&self, embeddings: Vec<(Chunk, Embedding)>)
            -> Result<(), <Self as EmbeddingStore>::ErrorType>;
    }
}
