use std::io::Error;

/// # Source
/// Trait for struct that allows reading the raw text for an external source
pub trait LoadSource {
    /// Called an returns a vector of raw text to generate embeddings for
    fn load(&self) -> Result<Vec<String>, Error>;
}

/// # Destination
/// Trait for struct that allows embeddings to be wrote to an external destination
pub trait EmbeddingStore {
    /// Takes an embedding and writes it to an external source
    fn store(&self, embedding: (String, Vec<f32>)) -> Result<(), Error>;
}

/// # VectorDBSource
/// Trait to query a vector database for similar entries
pub trait EmbeddingSource {
    /*
    need to do some digging on what the methods should look like here
     */
}
