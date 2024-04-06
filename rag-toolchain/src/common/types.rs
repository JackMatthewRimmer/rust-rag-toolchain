use serde::{Deserialize, Serialize};
use std::sync::Arc;

// ----------------- Embedding -----------------
/// # [`Embedding`]
/// The embedding type contains a vector and the associated
/// chunk of text and possibly some metadata. The type internally
/// uses [`Arc<T>`] to hold references to the internal values. This
/// makes it cheap to Clone and Copy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Embedding {
    /// The chunk that was used to generate the embedding
    chunk: Chunk,
    /// A vector of floats representing the embedding
    vector: Arc<[f32]>,
}

impl Embedding {
    /// # [`Embedding::new`]
    ///
    /// # Arguments
    /// * chunk: [`Chunk`] - the chunk associated with the embedding
    /// * vector: [`Into<Arc<[f32]>>`] - pointer to the embedding
    ///
    /// # Returns
    /// * [`Embedding`] - a new Embedding
    pub fn new(chunk: Chunk, vector: impl Into<Arc<[f32]>>) -> Self {
        Self {
            chunk,
            vector: vector.into(),
        }
    }

    /// # [`Embedding::chunk`]
    /// Getter for the [`Chunk`]
    ///
    /// # Returns
    /// * &[`Chunk`] - reference to the chunk
    pub fn chunk(&self) -> &Chunk {
        &self.chunk
    }

    /// # [`Embedding::vector`]
    /// Getter for the [`Vec<f32>`] vector
    ///
    /// # Returns
    /// * [`Vec<f32>`] - a copy of the vector
    pub fn vector(&self) -> Vec<f32> {
        self.vector.as_ref().to_vec()
    }
}
// ---------------------------------------------

// ----------------- Chunk ------------------
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Eq)]
/// # [`Chunk`]
/// A chunk is a piece of text with associated metadata. This is
/// type uses [`Arc<T>`] to hold references to the internal values.
/// so it is cheap to Clone and Copy.
pub struct Chunk {
    /// This is the text content
    content: Arc<str>,
    /// Any metadata associated with the chunk such as a date, author, etc.
    metadata: Arc<serde_json::Value>,
}

impl Chunk {
    /// # [`Chunk::new`]
    /// This is the constructor to use when we have some text with no metadata that
    /// we wish to include with it.
    /// # Arguments
    /// * content: [`Into<Arc<str>>`] - this is the text content of the chunk
    ///
    /// # Returns
    /// * [`Chunk`] - a new Chunk with no metadata
    pub fn new(chunk: impl Into<Arc<str>>) -> Self {
        Self {
            content: chunk.into(),
            metadata: Arc::new(serde_json::Value::Null),
        }
    }

    /// # [`Chunk::new_with_metadata`]
    /// This is the constructor to use when we have some text with metadata.
    /// Note the metadata does not influence any generated embeddings. It can just
    /// be kept with the text and embedding in whatever vector store you choose to use.
    ///
    /// # Arguments
    /// * content: [`Into<Arc<str>>`] - pointer to the chunk str
    /// * metadata: [`serde_json::Value`] - metadata associated with the chunk
    ///
    /// # Returns
    /// * [`Chunk`] - a new Chunk
    pub fn new_with_metadata(content: impl Into<Arc<str>>, metadata: serde_json::Value) -> Self {
        Self {
            content: content.into(),
            metadata: Arc::new(metadata),
        }
    }

    /// # [`Chunk::content`]
    /// Getter for the text content.
    ///
    /// # Returns
    /// * &[`str`] - reference to the chunk str
    pub fn content(&self) -> &str {
        &self.content
    }

    /// # [`Chunk::metadata`]
    /// Getter for the metadata
    /// # Returns
    /// * &[`serde_json::Value`] - reference to metadata associated with the chunk
    pub fn metadata(&self) -> &serde_json::Value {
        &self.metadata
    }
}
// ------------------------------------------

// ----------------- Chunks -----------------
/// Type alias for a vector of [`Chunk`]
pub type Chunks = Vec<Chunk>;
// -----------------------------------------
