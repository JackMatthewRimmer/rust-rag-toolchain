use crate::common::{Chunk, Chunks, EmbeddingModel, EmbeddingModelMetadata, TokenizerWrapper};
use std::num::NonZeroUsize;
use thiserror::Error;

use super::traits::Chunker;

/// # [`TokenChunker`]
/// This struct allows you to do fixed size chunking based on the number
/// of tokens in each chunk. We build around specific embedding models and based
/// on which embedding model is being used we will use the correlating tokenizer.
///
/// # Examples
/// ```
/// use rag_toolchain::chunkers::*;
/// use rag_toolchain::common::*;
/// use std::num::NonZeroUsize;
///
/// fn generate_chunks() {
///     let raw_text: &str = "This is a test string";
///     let window_size: usize = 1;
///     let chunk_size: NonZeroUsize = NonZeroUsize::new(2).unwrap();
///
///     const EMBEDDING_MODEL: OpenAIEmbeddingModel = OpenAIEmbeddingModel::TextEmbedding3Small;
///
///     let chunker: TokenChunker = TokenChunker::try_new(
///         chunk_size,
///         window_size,
///         EMBEDDING_MODEL,
///     )
///     .unwrap();
///
///     let chunks: Chunks = chunker.generate_chunks(raw_text).unwrap();
/// }
///
/// ```
pub struct TokenChunker {
    /// chunk_size: The size in tokens of each chunk
    chunk_size: NonZeroUsize,
    /// chunk_overlap: The number of tokens that overlap between each chunk
    chunk_overlap: usize,
    /// tokenizer: The type of tokenizer
    tokenizer: Box<dyn TokenizerWrapper>,
}

impl TokenChunker {
    /// # [`TokenChunker::try_new`]
    ///
    /// # Arguments
    /// * `chunk_size`: [`NonZeroUsize`] - The size in tokens of each chunk
    /// * `chunk_overlap`: [`usize`] - The number of tokens that overlap between each chunk
    /// * `embedding_model`: impl [`EmbeddingModel`] - The embedding model to use, this tells us what tokenizer
    ///                      to use
    ///
    /// # Errors
    /// * [`ChunkingError::InvalidChunkSize`] - Chunk size must be smaller than the maximum number of tokens
    /// * [`ChunkingError::ChunkOverlapTooLarge`] - Chunk overlap must be smaller than chunk size
    ///
    /// # Returns
    /// * [`TokenChunker`] - The token chunker
    pub fn try_new(
        chunk_size: NonZeroUsize,
        chunk_overlap: usize,
        embedding_model: impl EmbeddingModel,
    ) -> Result<Self, TokenChunkingError> {
        let metadata: EmbeddingModelMetadata = embedding_model.metadata();
        Self::validate_arguments(chunk_size.into(), chunk_overlap, metadata.max_tokens)?;
        let chunker = TokenChunker {
            chunk_size,
            chunk_overlap,
            tokenizer: metadata.tokenizer,
        };
        Ok(chunker)
    }

    // # [`TokenChunker::validate_arguments`]
    // function to validate arguments when [`TokenChunker::new`] is called
    //
    // # Arguments
    // * `chunk_size`: [`usize`] - The size in tokens of each chunk
    // * `chunk_overlap`: [`usize`] - The number of tokens that overlap between each chunk
    // * `max_chunk_size`: [`usize`] - The maximum number of tokens allowed which is defined
    //                      by the embedding model passed to the [`TokenChunker::new`] function
    // # Errors
    // * [`ChunkingError::InvalidChunkSize`] - Chunk size must be smaller than the maximum number of tokens
    // * [`ChunkingError::ChunkOverlapTooLarge`] - Chunk overlap must be smaller than chunk size
    //
    // # Returns
    // [`()`] - Result indicating whether the arguments are valid
    fn validate_arguments(
        chunk_size: usize,
        chunk_overlap: usize,
        max_chunk_size: usize,
    ) -> Result<(), TokenChunkingError> {
        if chunk_size > max_chunk_size {
            Err(TokenChunkingError::InvalidChunkSize(format!(
                "Chunk size must be smaller than {}",
                max_chunk_size
            )))?
        }

        if chunk_overlap >= chunk_size {
            Err(TokenChunkingError::ChunkOverlapTooLarge(
                "Window size must be smaller than chunk size".to_string(),
            ))?
        }
        Ok(())
    }
}

impl Chunker for TokenChunker {
    type ErrorType = TokenChunkingError;
    /// # [`TokenChunker::generate_chunks`]
    /// function to generate chunks from raw text
    ///
    /// # Arguments
    /// * `raw_text`: &[`str`] - The raw text to generate chunks from
    ///
    /// # Errors
    /// * [`ChunkingError::TokenizationError`] - Unable to tokenize text
    ///
    /// # Returns
    /// [`Chunks`] - The generated chunks
    fn generate_chunks(&self, raw_text: &str) -> Result<Chunks, Self::ErrorType> {
        // Generate token array from raw text
        let tokens: Vec<String> = self.tokenizer.tokenize(raw_text).ok_or_else(|| {
            TokenChunkingError::TokenizationError("Unable to tokenize text".to_string())
        })?;

        let chunk_size: usize = self.chunk_size.into();
        let mut chunks: Chunks = Chunks::new();

        let mut i = 0;
        while i < tokens.len() {
            let end = std::cmp::min(i + chunk_size, tokens.len());
            let chunk: Chunk = Chunk::new(tokens[i..end].to_vec().join("").trim());
            chunks.push(chunk);
            i += chunk_size - self.chunk_overlap;
        }

        Ok(Chunks::from(chunks))
    }
}

/// # [`ChunkingError`]
/// Custom error type representing errors that can occur during chunking
#[derive(Error, Debug, PartialEq, Eq)]
pub enum TokenChunkingError {
    #[error("{0}")]
    ChunkOverlapTooLarge(String),
    #[error("{0}")]
    TokenizationError(String),
    #[error("{0}")]
    InvalidChunkSize(String),
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::common::OpenAIEmbeddingModel::TextEmbeddingAda002;

    #[test]
    fn test_generate_chunks_with_valid_input() {
        let raw_text: &str = "This is a test string";
        let window_size: usize = 1;
        let chunk_size: NonZeroUsize = NonZeroUsize::new(2).unwrap();
        let chunker: TokenChunker =
            TokenChunker::try_new(chunk_size, window_size, TextEmbeddingAda002).unwrap();
        let chunks: Chunks = chunker.generate_chunks(raw_text).unwrap();
        let chunks: Vec<String> = chunks
            .into_iter()
            .map(|chunk| chunk.content().to_string())
            .collect();
        assert_eq!(chunks.len(), 5);
        assert_eq!(
            chunks,
            vec!["This is", "is a", "a test", "test string", "string"]
        );
    }

    #[test]
    fn test_generate_chunks_with_empty_string() {
        let raw_text: &str = "";
        let window_size: usize = 1;
        let chunk_size: NonZeroUsize = NonZeroUsize::new(2).unwrap();
        let chunker: TokenChunker =
            TokenChunker::try_new(chunk_size, window_size, TextEmbeddingAda002).unwrap();
        let chunks: Chunks = chunker.generate_chunks(raw_text).unwrap();
        let chunks: Vec<String> = chunks
            .into_iter()
            .map(|chunk| chunk.content().to_string())
            .collect();
        assert_eq!(chunks.len(), 0);
        assert_eq!(chunks, Vec::<String>::new());
    }

    #[test]
    fn test_generate_chunks_with_invalid_window_size() {
        let window_size: usize = 3;
        let chunk_size: NonZeroUsize = NonZeroUsize::new(2).unwrap();
        assert!(TokenChunker::try_new(chunk_size, window_size, TextEmbeddingAda002).is_err());
    }

    #[test]
    fn test_generate_chunks_with_invalid_chunk_size() {
        let window_size: usize = 3;
        let chunk_size: NonZeroUsize = NonZeroUsize::new(20000).unwrap();
        assert!(TokenChunker::try_new(chunk_size, window_size, TextEmbeddingAda002).is_err());
    }
}
