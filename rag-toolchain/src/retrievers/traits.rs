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
    type ErrorType: Error;
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
