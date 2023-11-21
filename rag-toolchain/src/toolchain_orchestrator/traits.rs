use std::io::Error;

/// # Source
/// Trait for struct that allows reading the raw text for an external source
pub trait EmbeddingDataSource {
    /// Called an returns a vector of raw text to generate embeddings for
    fn read_source_data(&self) -> Result<Vec<String>, Error>;
}

/// # Destination
/// Trait for struct that allows embeddings to be wrote to an external destination
pub trait EmbeddingDestination {
    /// Takes an embedding and writes it to an external source
    fn write_embedding(&self, embedding: (String, Vec<f32>)) -> Result<(), Error>;
}

/// # VectorDBSource
/// Trait to query a vector database for similar entries
pub trait VectorDBSource {
    /*
    need to do some digging on what the methods should look like here
     */
}

/// # EmbeddingClient
/// Trait for struct that allows embeddings to be generated
pub trait EmbeddingClient {
    fn generate_embeddings(&self) -> Result<Vec<f32>, Error>;
}
