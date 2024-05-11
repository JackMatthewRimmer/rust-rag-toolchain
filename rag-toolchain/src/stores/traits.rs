use crate::common::Embedding;
use std::error::Error;
use std::future::Future;

/// # [`EmbeddingStore`]
/// This is the trait defined for abstracting storing embeddings
/// into a vector database.
pub trait EmbeddingStore {
    /// The custom error type for the store
    type ErrorType: Error;
    /// # [`EmbeddingStore::store`]
    /// This method is used to store a single embedding in the store.
    ///
    /// # Arguments
    /// * `embedding`: [`Embedding`] - The embedding to store
    ///
    /// # Errors
    /// * [`Self::ErrorType`] - If the operation failed.
    ///
    /// # Returns
    /// * [`()`] - indicating success
    fn store(
        &self,
        embedding: Embedding,
    ) -> impl Future<Output = Result<(), Self::ErrorType>> + Send;
    fn store_batch(
        &self,
        embeddings: Vec<Embedding>,
    ) -> impl Future<Output = Result<(), Self::ErrorType>> + Send;
}
