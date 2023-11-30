use crate::toolchain_embeddings::embedding_models::{
    EmbeddingModelMetadata, HasMetadata, TokenizerWrapper,
};
use std::num::NonZeroUsize;

/// # ChunkingError
/// Custom error type representing errors that can occur during chunking
#[derive(Debug, PartialEq, Eq)]
pub enum ChunkingError {
    WindowSizeTooLarge(String),
    TokenizationError(String),
    InvalidChunkSize(String),
}

pub struct TokenChunker {
    chunk_size: NonZeroUsize,
    chunk_overlap: usize,
    tokenizer: Box<dyn TokenizerWrapper>,
}

impl TokenChunker {
    pub fn new(
        chunk_size: NonZeroUsize,
        chunk_overlap: usize,
        embedding_model: impl HasMetadata,
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

    /// # validate_arguments
    /// Responsible for checking if the given arguments are valid
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
            Err(ChunkingError::WindowSizeTooLarge(
                "Window size must be smaller than chunk size".to_string(),
            ))?
        }
        Ok(())
    }

    /// # generate_chunks
    /// function to generate chunks from raw text
    pub fn generate_chunks(&self, raw_text: &str) -> Result<Vec<String>, ChunkingError> {
        // Generate token array from raw text
        let tokens: Vec<String> = self.tokenizer.tokenize(raw_text).ok_or_else(|| {
            ChunkingError::TokenizationError("Unable to tokenize text".to_string())
        })?;

        let chunk_size: usize = self.chunk_size.into();
        let mut chunks = Vec::new();
        let mut i = 0;
        while i < tokens.len() {
            let end = std::cmp::min(i + chunk_size, tokens.len());
            let chunk: String = tokens[i..end].to_vec().join("").trim().to_string();
            chunks.push(chunk);
            i += chunk_size - self.chunk_overlap;
        }
        Ok(chunks)
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::toolchain_embeddings::embedding_models::OpenAIEmbeddingModel::TextEmbeddingAda002;

    #[test]
    fn test_generate_chunks_with_valid_input() {
        let raw_text: &str = "This is a test string";
        let window_size: usize = 1;
        let chunk_size: NonZeroUsize = NonZeroUsize::new(2).unwrap();
        let chunker: TokenChunker =
            TokenChunker::new(chunk_size, window_size, TextEmbeddingAda002).unwrap();
        let chunks: Vec<String> = chunker.generate_chunks(raw_text).unwrap();
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
        let chunks: Vec<String> = chunker.generate_chunks(raw_text).unwrap();
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
            ChunkingError::WindowSizeTooLarge(
                "Window size must be smaller than chunk size".to_string()
            )
        );
    }
}
