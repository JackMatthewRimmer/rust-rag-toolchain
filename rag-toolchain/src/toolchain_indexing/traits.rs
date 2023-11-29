use async_trait::async_trait;
use std::error::Error;
use std::io::Error as StdError;

/// # Source
/// Trait for struct that allows reading the raw text for an external source
pub trait LoadSource {
    /// Called an returns a vector of raw text to generate embeddings for
    fn load(&self) -> Result<Vec<String>, StdError>;
}

/// # Destination
/// Trait for struct that allows embeddings to be wrote to an external destination
#[async_trait]
pub trait EmbeddingStore {
    /// Takes an embedding and writes it to an external source
    async fn store(&self, embedding: (String, Vec<f32>)) -> Result<(), Box<dyn Error>>;
    async fn store_batch(&self, embeddings: Vec<(String, Vec<f32>)>) -> Result<(), Box<dyn Error>>;
}

pub trait StoreError {}
impl<T> From<T> for Box<dyn StoreError>
where
    T: StoreError + 'static,
{
    fn from(error: T) -> Self {
        Box::new(error)
    }
}

/// # VectorDBSource
/// Trait to query a vector database for similar entries
pub trait EmbeddingSource {
    /*
    need to do some digging on what the methods should look like here
     */
}
