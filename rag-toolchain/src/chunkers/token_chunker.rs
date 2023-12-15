use crate::common::embedding_shared::{EmbeddingModel, EmbeddingModelMetadata, TokenizerWrapper};
use crate::common::types::{Chunk, Chunks};
use std::num::NonZeroUsize;

/// # TokenChunker
/// Allows you to chunk text using token based chunking
pub struct TokenChunker {
    /// chunk_size: The size in tokens of each chunk
    chunk_size: NonZeroUsize,
    /// chunk_overlap: The number of tokens that overlap between each chunk
    chunk_overlap: usize,
    /// tokenizer: The type of tokenizer
    tokenizer: Box<dyn TokenizerWrapper>,
}

impl TokenChunker {
    /// # new
    ///
    /// # Arguments
    /// * `chunk_size` - The size in tokens of each chunk
    /// * `chunk_overlap` - The number of tokens that overlap between each chunk
    /// * `embedding_model` - The embedding model to use
    ///
    /// # Errors
    /// * [`ChunkingError::InvalidChunkSize`] - Chunk size must be smaller than the maximum number of tokens
    /// * [`ChunkingError::ChunkOverlapTooLarge`] - Chunk overlap must be smaller than chunk size
    ///
    /// # Returns
    /// * `Result<TokenChunker, ChunkingError>` - The token chunker
    pub fn new(
        chunk_size: NonZeroUsize,
        chunk_overlap: usize,
        embedding_model: impl EmbeddingModel,
    ) -> Result<Self, ChunkingError> {
        let metadata: EmbeddingModelMetadata = embedding_model.metadata();
        Self::validate_arguments(chunk_size.into(), chunk_overlap, metadata.max_tokens)?;
        let chunker = TokenChunker {
            chunk_size,
            chunk_overlap,
            tokenizer: metadata.tokenizer,
        };
        Ok(chunker)
    }

    // # validate_arguments
    // function to validate arguments when [`TokenChunker::new`] is called
    //
    // # Arguments
    // * `chunk_size` - The size in tokens of each chunk
    // * `chunk_overlap` - The number of tokens that overlap between each chunk
    // * `max_chunk_size` - The maximum number of tokens allowed which is defined
    //                      by the embedding model passed to the [`TokenChunker::new`] function
    // # Errors
    // * [`ChunkingError::InvalidChunkSize`] - Chunk size must be smaller than the maximum number of tokens
    // * [`ChunkingError::ChunkOverlapTooLarge`] - Chunk overlap must be smaller than chunk size
    //
    // # Returns
    // `Result<(), ChunkingError>` - Result indicating whether the arguments are valid
    fn validate_arguments(
        chunk_size: usize,
        chunk_overlap: usize,
        max_chunk_size: usize,
    ) -> Result<(), ChunkingError> {
        if chunk_size > max_chunk_size {
            Err(ChunkingError::InvalidChunkSize(format!(
                "Chunk size must be smaller than {}",
                max_chunk_size
            )))?
        }

        if chunk_overlap >= chunk_size {
            Err(ChunkingError::ChunkOverlapTooLarge(
                "Window size must be smaller than chunk size".to_string(),
            ))?
        }
        Ok(())
    }

    /// # generate_chunks
    /// function to generate chunks from raw text
    ///
    /// # Arguments
    /// * `raw_text` - The raw text to generate chunks from
    ///
    /// # Errors
    /// * [`ChunkingError::TokenizationError`] - Unable to tokenize text
    ///
    /// # Returns
    /// `Result<Chunks, ChunkingError>` - The generated chunks
    pub fn generate_chunks(&self, raw_text: &str) -> Result<Chunks, ChunkingError> {
        // Generate token array from raw text
        let tokens: Vec<String> = self.tokenizer.tokenize(raw_text).ok_or_else(|| {
            ChunkingError::TokenizationError("Unable to tokenize text".to_string())
        })?;

        let chunk_size: usize = self.chunk_size.into();
        let mut chunks: Vec<Chunk> = Vec::new();

        let mut i = 0;
        while i < tokens.len() {
            let end = std::cmp::min(i + chunk_size, tokens.len());
            let chunk: Chunk = Chunk::from(tokens[i..end].to_vec().join("").trim());
            chunks.push(chunk);
            i += chunk_size - self.chunk_overlap;
        }

        Ok(Chunks::from(chunks))
    }
}

/// # ChunkingError
/// Custom error type representing errors that can occur during chunking
#[derive(Debug, PartialEq, Eq)]
pub enum ChunkingError {
    ChunkOverlapTooLarge(String),
    TokenizationError(String),
    InvalidChunkSize(String),
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::common::embedding_shared::OpenAIEmbeddingModel::TextEmbeddingAda002;

    #[test]
    fn test_generate_chunks_with_valid_input() {
        let raw_text: &str = "This is a test string";
        let window_size: usize = 1;
        let chunk_size: NonZeroUsize = NonZeroUsize::new(2).unwrap();
        let chunker: TokenChunker =
            TokenChunker::new(chunk_size, window_size, TextEmbeddingAda002).unwrap();
        let chunks: Chunks = chunker.generate_chunks(raw_text).unwrap();
        let chunks: Vec<String> = chunks.into_iter().map(|chunk| chunk.into()).collect();
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
            TokenChunker::new(chunk_size, window_size, TextEmbeddingAda002).unwrap();
        let chunks: Chunks = chunker.generate_chunks(raw_text).unwrap();
        let chunks: Vec<String> = chunks.into_iter().map(|chunk| chunk.into()).collect();
        assert_eq!(chunks.len(), 0);
        assert_eq!(chunks, Vec::<String>::new());
    }

    #[test]
    fn test_generate_chunks_with_invalid_arguments() {
        let window_size: usize = 3;
        let chunk_size: NonZeroUsize = NonZeroUsize::new(2).unwrap();
        let chunker: ChunkingError =
            match TokenChunker::new(chunk_size, window_size, TextEmbeddingAda002) {
                Ok(_) => panic!("Expected error"),
                Err(e) => e,
            };

        assert_eq!(
            chunker,
            ChunkingError::ChunkOverlapTooLarge(
                "Window size must be smaller than chunk size".to_string()
            )
        );
    }
}
