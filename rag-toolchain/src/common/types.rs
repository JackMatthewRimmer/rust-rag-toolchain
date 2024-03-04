use serde::{Deserialize, Serialize};
use std::sync::Arc;

// ----------------- Embedding -----------------
/// # [`Embedding`]
/// Custom type that wraps a pointer to an embedding/vector.
/// It is immutable and thread safe
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Embedding {
    chunk: Chunk,
    vector: Arc<[f32]>,
}

impl Embedding {
    /// # [`Embedding::new`]
    ///
    /// # Arguments
    /// * `Chunk` - the chunk associated with the embedding
    /// * `impl Into<Arc<[f32]>>` - pointer to the embedding
    ///
    /// # Returns
    /// * [`Embedding`] - a new Embedding
    pub fn new(chunk: Chunk, vector: impl Into<Arc<[f32]>>) -> Self {
        Self {
            chunk,
            vector: vector.into(),
        }
    }

    pub fn chunk(&self) -> &Chunk {
        &self.chunk
    }

    pub fn vector(&self) -> Vec<f32> {
        self.vector.as_ref().to_vec()
    }
}
// ---------------------------------------------

// ----------------- Chunk ------------------
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Eq)]
/// # [`Chunk`]
/// Custom type that wraps a pointer to a piece of text.
/// and also some metadata associated with the text.
pub struct Chunk {
    content: Arc<str>,
    metadata: Arc<serde_json::Value>,
}

impl Chunk {
    /// # [`Chunk::new`]
    /// * `impl Into<Arc<str>>` - this is the text content of the chunk
    ///
    /// # Returns
    /// * [`Chunk`] - a new Chunk
    pub fn new(chunk: impl Into<Arc<str>>) -> Self {
        Self {
            content: chunk.into(),
            metadata: Arc::new(serde_json::Value::Null),
        }
    }

    /// # [`Chunk::new_with_metadata`]
    ///
    /// # Arguments
    /// * `Arc<str>` - pointer to the chunk str
    /// * `serde_json::Value` - metadata associated with the chunk
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
    ///
    /// # Returns
    /// * [`&str`] - pointer to the chunk str
    pub fn content(&self) -> &str {
        &self.content
    }

    /// # [`Chunk::metadata`]
    ///
    /// # Returns
    /// * [`&serde_json::Value`] - metadata associated with the chunk
    pub fn metadata(&self) -> &serde_json::Value {
        &self.metadata
    }
}
// ------------------------------------------

// ----------------- Chunks -----------------
/// Type alias for a vector of [`Chunk`]
pub type Chunks = Vec<Chunk>;
// -----------------------------------------
