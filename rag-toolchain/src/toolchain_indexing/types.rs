use std::rc::Rc;

// ----------------- Embedding -----------------
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

    pub fn into_vec<T>(iter: impl Iterator<Item = T>) -> Vec<Self>
    where
        T: Into<Self>,
    {
        let mut vec: Vec<Embedding> = Vec::new();
        for item in iter {
            vec.push(item.into());
        }
        vec
    }
}

impl From<Vec<f32>> for Embedding {
    fn from(embedding: Vec<f32>) -> Self {
        Self {
            embedding: embedding.into(),
        }
    }
}

// ---------------------------------------------

// ----------------- Chunk ------------------
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

impl Clone for Chunk {
    fn clone(&self) -> Self {
        Self {
            chunk: self.chunk.clone(),
        }
    }
}

impl From<String> for Chunk {
    fn from(string: String) -> Self {
        Self {
            chunk: string.into(),
        }
    }
}

impl From<&str> for Chunk {
    fn from(string: &str) -> Self {
        Self {
            chunk: string.into(),
        }
    }
}

impl From<Chunk> for String {
    fn from(chunk: Chunk) -> Self {
        (*chunk.chunk).to_string()
    }
}
// ------------------------------------------

// ----------------- Chunks -----------------
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

impl From<Vec<Chunk>> for Chunks {
    fn from(chunks: Vec<Chunk>) -> Self {
        Self {
            chunks: chunks.into(),
        }
    }
}
// -----------------------------------------
