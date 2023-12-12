use std::error::Error;

/// # Source
/// Trait for struct that allows reading the raw text for an external source
pub trait LoadSource {
    type ErrorType: Error;
    /// Called an returns a vector of raw text to generate embeddings for
    fn load(&self) -> Result<Vec<String>, Self::ErrorType>;
}
