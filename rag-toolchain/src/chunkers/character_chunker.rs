use crate::chunkers::{Chunker, StreamedChunker};
use crate::common::{Chunk, Chunks};
use std::convert::Infallible;
use std::error::Error;
use std::fmt::Display;
use std::num::NonZeroUsize;

pub struct CharacterChunker {
    /// chunk_size: the number of characters in each chunk
    chunk_size: NonZeroUsize,
    /// chunk_overlap: the number over characters
    /// shared between neighbouring chunks
    chunk_overlap: usize,
}

impl CharacterChunker {
    /// [`TokenChunker::new`]
    ///
    /// # Arguements
    /// * `chunk_size`: [`NonZeroUsize`] - The number of characters in each chunk
    /// * `chunk_overlap`: [`usize`] - The number of characters shared between
    ///                   neighbouring chunks
    /// # Returns
    /// [`TokenChunker`]
    // TODO: needs to be try new and validated like the TokenChunker
    pub fn new(chunk_size: NonZeroUsize, chunk_overlap: usize) -> Self {
        Self {
            chunk_size,
            chunk_overlap,
        }
    }
}

impl Chunker for CharacterChunker {
    type ErrorType = Infallible;
    fn generate_chunks(&self, raw_text: &str) -> Result<Chunks, Self::ErrorType> {
        let mut chunks: Chunks = Vec::new();
        let chunk_size: usize = self.chunk_size.into();

        let mut i = 0;
        while i < raw_text.len() {
            let end = std::cmp::min(i + chunk_size, raw_text.len());
            let chunk: Chunk = Chunk::new(&raw_text[i..end]);
            chunks.push(chunk);
            i += chunk_size - self.chunk_overlap;
        }

        Ok(chunks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // TODO: will need test cases when the constructor becomes try_new.
    // This is because we need to validate the arguements

    #[test]
    fn test_generate_chunks_with_valid_input() {
        let raw_text: &str = "This is a test string";
        let chunk_overlap: usize = 1;
        let chunk_size: NonZeroUsize = NonZeroUsize::new(2).unwrap();
        let chunker: CharacterChunker = CharacterChunker::new(chunk_size, chunk_overlap);
        let chunks = chunker.generate_chunks(raw_text).unwrap();
        let chunk_strings: Vec<String> = chunks
            .into_iter()
            .map(|chunk| chunk.content().to_string())
            .collect();
        assert_eq!(
            chunk_strings,
            vec![
                "Th", "hi", "is", "s ", " i", "is", "s ", " a", "a ", " t", "te", "es", "st", "t ",
                " s", "st", "tr", "ri", "in", "ng", "g"
            ]
        );
    }
}
