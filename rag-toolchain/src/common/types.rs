use std::sync::Arc;

// ----------------- Embedding -----------------
/// # Embedding
/// Custom type that wraps a pointer to an embedding
/// immutable and thread safe
/// this will be returned by an Embedding Generator
#[derive(Debug, Clone, PartialEq)]
pub struct Embedding {
    embedding: Arc<[f32]>,
}

impl Embedding {
    /// # new
    ///
    /// # Arguments
    /// * `Arc<[f32]>` - pointer to the embedding
    ///
    /// # Returns
    /// * `Embedding` - a new Embedding
    pub fn new(embedding: Arc<[f32]>) -> Self {
        Self { embedding }
    }

    /// # embedding
    ///
    /// # Returns
    /// * `Arc<[f32]>` - pointer to the embedding
    pub fn embedding(&self) -> Arc<[f32]> {
        Arc::clone(&self.embedding)
    }

    /// # iter_to_vec
    ///
    /// Helper function to generate a vector given an iterator of items
    /// that can be converted into an Embedding
    ///
    /// # Arguments
    /// * `Iterator<Item = T>` - iterator of items where T: Into<Embedding>
    ///
    /// # Returns
    /// * `Vec<Embedding>` - vector of Embeddings
    pub fn iter_to_vec<T>(iter: impl Iterator<Item = T>) -> Vec<Self>
    where
        T: Into<Self>,
    {
        let mut embedding: Vec<Self> = Vec::new();
        for item in iter {
            embedding.push(item.into());
        }
        embedding
    }

    /// # from_vec
    ///
    /// Helper function to convert a vector of items into a vector
    /// of embeddings
    ///
    /// # Arguments
    /// * `Vec<T>` - vector of items where T: Into<Embedding>
    ///
    /// # Returns
    /// * `Vec<Embedding>` - vector of Embeddings
    pub fn from_vec<T>(vec: Vec<T>) -> Vec<Self>
    where
        T: Into<Self>,
    {
        let mut embedding: Vec<Self> = Vec::new();
        for item in vec.into_iter() {
            embedding.push(item.into());
        }
        embedding
    }
}

impl<T> From<T> for Embedding
where
    T: Into<Arc<[f32]>>,
{
    fn from(embedding: T) -> Self {
        Self {
            embedding: embedding.into(),
        }
    }
}

impl From<Embedding> for Vec<f32> {
    fn from(embedding: Embedding) -> Self {
        embedding.embedding().to_vec()
    }
}
// ---------------------------------------------

// ----------------- Chunk ------------------
#[derive(Debug, Clone, PartialEq)]
/// # Chunk
/// Custom type that wraps a pointer to a text chunk
/// this will be returned by a form of Chunker struct
/// immutable and thread safe
pub struct Chunk {
    chunk: Arc<str>,
}

impl Chunk {
    /// # new
    ///
    /// # Arguments
    /// * `Arc<str>` - pointer to the chunk str
    ///
    /// # Returns
    /// * `Chunk` - a new Chunk
    pub fn new(chunk: Arc<str>) -> Self {
        Self { chunk }
    }

    /// # chunk
    ///
    /// # Returns
    /// * `Arc<str>` - pointer to the chunk str
    pub fn chunk(&self) -> Arc<str> {
        Arc::clone(&self.chunk)
    }
}

impl<T> From<T> for Chunk
where
    T: Into<Arc<str>>,
{
    fn from(chunk: T) -> Self {
        Self {
            chunk: chunk.into(),
        }
    }
}

impl From<Chunk> for String {
    fn from(chunk: Chunk) -> Self {
        chunk.chunk().to_string()
    }
}
// ------------------------------------------

// ----------------- Chunks -----------------
pub type Chunks = Vec<Chunk>;
// -----------------------------------------
