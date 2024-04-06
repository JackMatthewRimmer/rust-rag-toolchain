use crate::common::Chunks;
use std::future::Future;
use std::{error::Error, num::NonZeroU32};

/// # [`AsyncRetriever`]
///
/// This trait is used to define structs that allow for searching of similar text.
/// The common workflow is to instaniate a store and then call .as_retriever()
/// to get a retriever for that vector database. Then you can call the methods
/// described by this trait to retrieve similar text.
pub trait AsyncRetriever {
    /// Custom error type for the retriever
    type ErrorType: Error;
    /// # [`AsyncRetriever::retrieve`]
    ///
    /// This method is used to retrieve similar text from the store.
    /// It takes input text which internally should be embedded before the
    /// vector search. Note this method returns a future so will be used from
    /// an async context.
    ///
    /// # Arguments
    /// * `text`: &[`str`] - The input text to search for similar text.
    /// * `top_k`: [`NonZeroU32`] - The number of similar text to return.
    ///
    /// # Errors
    /// * [`Self::ErrorType`] - If the operation failed.
    ///
    /// # Returns
    /// * [`Chunks`] - The most similar text to the input text.
    fn retrieve(
        &self,
        text: &str,
        top_k: NonZeroU32,
    ) -> impl Future<Output = Result<Chunks, Self::ErrorType>> + Send;
}

#[cfg(test)]
use mockall::*;
#[cfg(test)]
mock! {
    pub AsyncRetriever {}
    impl AsyncRetriever for AsyncRetriever {
        type ErrorType = std::io::Error;
        async fn retrieve(&self, text: &str, top_k: NonZeroU32) -> Result<Chunks, <Self as AsyncRetriever>::ErrorType>;
    }
}
