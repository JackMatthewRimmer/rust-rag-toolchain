use crate::common::Chunk;
use std::{error::Error, num::NonZeroU32};
use std::future::Future;

/*

There ideally needs to be a way to set similarity thresholds
Easily search for one without having to index and array
and search for multiple

Also would like to ensure

Probably best to just offer all of these as separate methods

 */

/// # [`AsyncRetriever`]
/// Trait for stores that allow for similar text to be retrieved using embeddings
pub trait AsyncRetriever {
    type ErrorType: Error;
    fn retrieve(&self, text: &str, top_k: NonZeroU32) -> impl Future<Output = Result<Vec<Chunk>, Self::ErrorType>> + Send;
}

#[cfg(test)]
use mockall::*;
#[cfg(test)]
mock! {
    pub AsyncRetriever {}
    impl AsyncRetriever for AsyncRetriever {
        type ErrorType = std::io::Error;
        async fn retrieve(&self, text: &str, top_k: NonZeroU32) -> Result<Vec<Chunk>, <Self as AsyncRetriever>::ErrorType>;
    }
}
