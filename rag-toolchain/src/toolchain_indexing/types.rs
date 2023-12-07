use std::rc::Rc;

// ----------------- Embedding -----------------
#[derive(Debug, Clone)]
pub struct Embedding {
    embedding: Rc<[f32]>,
}

impl Embedding {
    pub fn new(embedding: Rc<[f32]>) -> Self {
        Self { embedding }
    }

    pub fn embedding(&self) -> &Rc<[f32]> {
        &self.embedding
    }

    pub fn from_iter<T>(iter: impl Iterator<Item = T>) -> Vec<Self>
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

// Generic impl for any type that can be converted into a Rc<[f32]>
impl<T> From<T> for Embedding
where
    T: Into<Rc<[f32]>>,
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
    chunk: Rc<str>,
}

impl Chunk {
    pub fn new(chunk: Rc<str>) -> Self {
        Self { chunk }
    }

    pub fn chunk(&self) -> &Rc<str> {
        &self.chunk
    }
}

impl<T> From<T> for Chunk
where
    T: Into<Rc<str>>,
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
