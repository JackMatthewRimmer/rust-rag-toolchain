use std::rc::Rc;
use std::sync::Arc;

// ----------------- Embedding -----------------
#[derive(Debug, Clone)]
pub struct Embedding {
    embedding: Arc<[f32]>,
}

impl Embedding {
    pub fn new(embedding: Arc<[f32]>) -> Self {
        Self { embedding }
    }

    pub fn embedding(&self) -> Arc<[f32]> {
        Arc::clone(&self.embedding)
    }

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
#[derive(Debug, Clone)]
pub struct Chunk {
    chunk: Arc<str>,
}

impl Chunk {
    pub fn new(chunk: Arc<str>) -> Self {
        Self { chunk }
    }

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
#[derive(Debug, Clone)]
pub struct Chunks {
    chunks: Rc<[Chunk]>,
}

impl Chunks {
    pub fn new(chunks: Rc<[Chunk]>) -> Self {
        Self { chunks }
    }

    pub fn chunks(&self) -> &Rc<[Chunk]> {
        &self.chunks
    }

    pub fn to_vec<T>(&self) -> Vec<T>
    where
        T: From<Chunk>,
    {
        let mut vec: Vec<T> = Vec::new();
        for chunk in self.chunks.iter() {
            vec.push(T::from(chunk.clone()));
        }
        vec
    }
}

impl<T> From<T> for Chunks
where
    T: Into<Rc<[Chunk]>>,
{
    fn from(chunks: T) -> Self {
        Self {
            chunks: chunks.into(),
        }
    }
}

impl From<Chunks> for Vec<String> {
    fn from(chunks: Chunks) -> Self {
        chunks.to_vec()
    }
}
// -----------------------------------------
