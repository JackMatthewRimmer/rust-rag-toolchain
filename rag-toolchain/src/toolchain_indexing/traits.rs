use async_trait::async_trait;
use std::error::Error;
use std::io::Error as StdError;
use std::rc::Rc;

pub type Embedding = Rc<[f32]>;
// Once a chunk is made is should be immutable
pub type Chunk = Rc<str>;
// Chunks should be immutable
pub type Chunks = Rc<[Chunk]>;

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
    // The custom error type for the store
    type ErrorType: Error;
    /// Takes an embedding and writes it to an external source
    async fn store(&self, embedding: (String, Vec<f32>)) -> Result<(), Self::ErrorType>;
    async fn store_batch(&self, embeddings: Vec<(String, Vec<f32>)>)
        -> Result<(), Self::ErrorType>;
}

/// # VectorDBSource
/// Trait to query a vector database for similar entries
pub trait EmbeddingSource {
    /*
    need to do some digging on what the methods should look like here
     */
}
