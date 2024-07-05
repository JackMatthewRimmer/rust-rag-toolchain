use crate::chunkers::Chunker;
use crate::common::{Chunk, Chunks};
use futures::{Stream, StreamExt};
use std::convert::Infallible;
use std::num::NonZeroUsize;

pub struct CharacterChunker {
    /// chunk_size: the number of characters in each chunk
    chunk_size: NonZeroUsize,
    /// chunk_overlap: the number over characters
    /// shared between neighbouring chunks
    chunk_overlap: usize,
}

impl CharacterChunker {
    /// [`TokenChunker::try_new`]
    ///
    /// # Arguements
    /// * `chunk_size`: [`NonZeroUsize`] - The number of characters in each chunk
    /// * `chunk_overlap`: [`usize`] - The number of characters shared between
    ///                   neighbouring chunks
    ///
    /// # Errors
    /// This function will error if you provide a chunk_overlap greater than or equal to
    /// the chunk_size.
    ///
    /// # Returns
    /// [`TokenChunker`]
    pub fn try_new(chunk_size: NonZeroUsize, chunk_overlap: usize) -> Result<Self, String> {
        if chunk_overlap >= chunk_size.into() {
            return Err("chunk_overlap cannot be greater than or equal to chunk_size".into());
        }

        Ok(Self {
            chunk_size,
            chunk_overlap,
        })
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

pub struct CharacterChunkStream {
    stream: Box<dyn Stream<Item = std::io::Result<u8>> + Unpin>,
    buffer: Vec<u8>,
    chunk_size: usize,
    chunk_overlap: usize,
}

impl CharacterChunkStream {
    fn handle_ready_poll(
        &mut self,
        poll_result: Option<std::io::Result<u8>>,
    ) -> std::task::Poll<Option<Result<Chunk, CharacterChunkStreamError>>> {
        return match poll_result {
            None => std::task::Poll::Ready(None),
            Some(result) => {
                let chunk = self.handle_result(result);
                return match chunk {
                    None => std::task::Poll::Pending,
                    Some(chunk) => std::task::Poll::Ready(Some(chunk)),
                };
            }
        };
    }

    fn handle_result(
        &mut self,
        result: std::io::Result<u8>,
    ) -> Option<Result<Chunk, CharacterChunkStreamError>> {
        return match result {
            Err(error) => Some(Err(CharacterChunkStreamError::IoError(error))),
            Ok(char) => {
                self.buffer.push(char);
                return match self.buffer.len() >= self.chunk_size {
                    true => Some(self.consume_buffer()),
                    false => None,
                };
            }
        };
    }

    fn consume_buffer(&mut self) -> Result<Chunk, CharacterChunkStreamError> {
        let string: &str = std::str::from_utf8(&self.buffer)
            .map_err(|error| CharacterChunkStreamError::Utf8Error(error))?;
        let len = self.buffer.len();
        let window = &self.buffer[len - self.chunk_overlap..len];
        let chunk = Chunk::new(string);
        self.buffer = Vec::from(window);
        Ok(chunk)
    }
}

impl Stream for CharacterChunkStream {
    type Item = Result<Chunk, CharacterChunkStreamError>;
    fn poll_next(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let mut_self = self.get_mut();
        let poll_result: std::task::Poll<_> = mut_self.stream.poll_next_unpin(cx);

        return match poll_result {
            std::task::Poll::Pending => std::task::Poll::Pending,
            std::task::Poll::Ready(ready_result) => mut_self.handle_ready_poll(ready_result),
        };
    }
}

pub enum CharacterChunkStreamError {
    IoError(std::io::Error),
    Utf8Error(std::str::Utf8Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_chunks_with_valid_input() {
        let raw_text: &str = "This is a test string";
        let chunk_overlap: usize = 1;
        let chunk_size: NonZeroUsize = NonZeroUsize::new(2).unwrap();
        let chunker: CharacterChunker =
            CharacterChunker::try_new(chunk_size, chunk_overlap).unwrap();
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

    #[test]
    fn test_generate_chunks_with_empty_string() {
        let raw_text: &str = "";
        let chunk_overlap: usize = 1;
        let chunk_size: NonZeroUsize = NonZeroUsize::new(2).unwrap();
        let chunker: CharacterChunker =
            CharacterChunker::try_new(chunk_size, chunk_overlap).unwrap();
        let chunks = chunker.generate_chunks(raw_text).unwrap();
        let chunk_strings: Vec<String> = chunks
            .into_iter()
            .map(|chunk| chunk.content().to_string())
            .collect();
        assert_eq!(chunk_strings, Vec::<String>::new());
    }

    #[test]
    fn test_try_new_with_invalid_arguments() {
        let chunk_overlap: usize = 3;
        let chunk_size: NonZeroUsize = NonZeroUsize::new(2).unwrap();
        assert!(CharacterChunker::try_new(chunk_size, chunk_overlap).is_err());

        let chunk_overlap: usize = 2;
        let chunk_size: NonZeroUsize = NonZeroUsize::new(2).unwrap();
        assert!(CharacterChunker::try_new(chunk_size, chunk_overlap).is_err())
    }
}
