use crate::common::types::Chunk;
use async_trait::async_trait;
use std::{error::Error, num::NonZeroU32};

/*

There ideally needs to be a way to set similarity thresholds
Easily search for one without having to index and array
and search for multiple

Also would like to ensure

Probably best to just offer all of these as separate methods

 */

/// # EmbeddingRetriever
/// Trait for stores that allow for similar text to be retrieved using embeddings
#[async_trait]
pub trait AsyncRetriever {
    type ErrorType: Error;
    async fn retrieve(&self, text: &str, top_k: NonZeroU32) -> Result<Vec<Chunk>, Self::ErrorType>;
}
