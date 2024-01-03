use crate::common::types::Chunk;
use async_trait::async_trait;
use std::{error::Error, num::NonZeroI16};

/// # EmbeddingRetriever
/// Trait for stores that allow for similar text to be retrieved using embeddings
#[async_trait]
pub trait AsyncRetriever {
    type ErrorType: Error;
    async fn retrieve(
        &self,
        text: &str,
        number_of_results: NonZeroI16,
    ) -> Result<Vec<Chunk>, Self::ErrorType>;
}
