use async_trait::async_trait;
use std::error::Error;

/// # [`LoadSource`]
/// Trait that allows reading the raw text for an external source
pub trait LoadSource {
    type ErrorType: Error;
    /// Called an returns a vector of raw text to generate embeddings for
    fn load(&self) -> Result<Vec<String>, Self::ErrorType>;
}

/// # [`AsyncLoadSource`]
/// Async version of [`LoadSource`] to support loading raw text from an external source
#[async_trait]
pub trait AsyncLoadSource {
    type ErrorType: Error;
    /// Called an returns a vector of raw text to generate embeddings for
    async fn load(&self) -> Result<Vec<String>, Self::ErrorType>;
}
