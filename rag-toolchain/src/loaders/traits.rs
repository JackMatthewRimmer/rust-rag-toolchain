use std::error::Error;
use std::future::Future;

/// # [`LoadSource`]
/// Trait that allows reading the raw text for an external source
pub trait LoadSource {
    type ErrorType: Error;
    /// Called an returns a vector of raw text to generate embeddings for
    fn load(&self) -> Result<Vec<String>, Self::ErrorType>;
}

/// # [`AsyncLoadSource`]
/// Async version of [`LoadSource`] to support loading raw text from an external source
pub trait AsyncLoadSource {
    type ErrorType: Error;
    /// Called an returns a vector of raw text to generate embeddings for
    fn load(&self) -> impl Future<Output = Result<Vec<String>, Self::ErrorType>> + Send;
}
