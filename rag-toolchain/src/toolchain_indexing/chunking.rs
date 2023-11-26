use crate::toolchain_embeddings::embedding_models::EmbeddingModels;
use std::num::NonZeroUsize;
use tiktoken_rs::tokenizer::Tokenizer;
use tiktoken_rs::CoreBPE;

/// # ChunkingError
/// Custom error type representing errors that can occur during chunking
#[derive(Debug, PartialEq, Eq)]
pub enum ChunkingError {
    WindowSizeTooLarge(String),
    TokenizationError(String),
    InvalidChunkSize(String),
}

trait TokenizerWrapper {
    fn tokenize(&self, text: &str) -> Result<Vec<String>, ChunkingError>;
}

struct OpenAITokenizer {
    bpe: CoreBPE,
}

impl OpenAITokenizer {
    pub fn new(model: Tokenizer) -> Self {
        OpenAITokenizer {
            bpe: tiktoken_rs::get_bpe_from_tokenizer(model).unwrap(),
        }
    }
}

impl TokenizerWrapper for OpenAITokenizer {
    fn tokenize(&self, text: &str) -> Result<Vec<String>, ChunkingError> {
        // Generate token array from raw text
        let tokens = self
            .bpe
            .split_by_token(text, true)
            .map_err(|_| ChunkingError::TokenizationError("Failed to tokenize text".to_string()))?;
        return Ok(tokens);
    }
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
        embedding_model: EmbeddingModels,
    ) -> Result<Self, ChunkingError> {
        let (max_chunk_size, tokenizer) = Self::static_metadata(&embedding_model);
        Self::validate_arguments(chunk_size.into(), chunk_overlap, max_chunk_size)?;
        let dyn_tokenizer: Box<dyn TokenizerWrapper> = Box::new(tokenizer);
        let chunker = TokenChunker {
            chunk_size,
            chunk_overlap,
            tokenizer: dyn_tokenizer,
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
            return Err(ChunkingError::InvalidChunkSize(format!(
                "Chunk size must be smaller than {}",
                max_chunk_size
            )));
        }

        if chunk_overlap >= chunk_size {
            return Err(ChunkingError::WindowSizeTooLarge(
                "Window size must be smaller than chunk size".to_string(),
            ));
        }
        return Ok(());
    }

    /// # static_metadata
    /// Helper function to resolve the needed metadata for the given embedding model
    /// Returns an impl of TokenizerWrapper and the max chunk size
    fn static_metadata(embedding_model: &EmbeddingModels) -> (usize, impl TokenizerWrapper) {
        match embedding_model {
            EmbeddingModels::OpenAI(model) => {
                let metadata = model.metadata();
                return (
                    metadata.max_tokens,
                    OpenAITokenizer::new(metadata.tokenizer),
                );
            }
        }
    }
    /// # generate_chunks
    /// function to generate chunks from raw text
    pub fn generate_chunks(&self, raw_text: &str) -> Result<Vec<String>, ChunkingError> {
        // Generate token array from raw text
        let tokens: Vec<String> = self.tokenizer.tokenize(raw_text)?;
        let chunk_size: usize = self.chunk_size.into();

        let mut chunks = Vec::new();
        let mut i = 0;
        while i < tokens.len() {
            let end = std::cmp::min(i + chunk_size, tokens.len());
            let chunk: String = tokens[i..end].to_vec().join("").trim().to_string();
            chunks.push(chunk);
            i += chunk_size - self.chunk_overlap;
        }
        return Ok(chunks);
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::toolchain_embeddings::embedding_models::{
        OpenAIEmbeddingModel, OpenAIEmbeddingModel::TextEmbeddingAda002,
    };

    #[test]
    fn test_generate_chunks_with_valid_input() {
        let raw_text: &str = "This is a test string";
        let window_size: usize = 1;
        let chunk_size: NonZeroUsize = NonZeroUsize::new(2).unwrap();
        let model: EmbeddingModels = OpenAIEmbeddingModel::new(TextEmbeddingAda002);
        let chunker: TokenChunker = TokenChunker::new(chunk_size, window_size, model).unwrap();
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
        let model: EmbeddingModels = OpenAIEmbeddingModel::new(TextEmbeddingAda002);
        let chunker: TokenChunker = TokenChunker::new(chunk_size, window_size, model).unwrap();
        let chunks: Vec<String> = chunker.generate_chunks(raw_text).unwrap();
        assert_eq!(chunks.len(), 0);
        assert_eq!(chunks, Vec::<String>::new());
    }

    #[test]
    fn test_generate_chunks_with_invalid_arguments() {
        let raw_text: &str = "This is a test string";
        let window_size: usize = 3;
        let chunk_size: NonZeroUsize = NonZeroUsize::new(2).unwrap();
        let model: EmbeddingModels = OpenAIEmbeddingModel::new(TextEmbeddingAda002);
        let chunker: TokenChunker = TokenChunker::new(chunk_size, window_size, model).unwrap();
        let chunks: ChunkingError = chunker
            .generate_chunks(raw_text)
            .expect_err("Failed to generate chunks");
        assert_eq!(
            chunks,
            ChunkingError::WindowSizeTooLarge(
                "Window size must be smaller than chunk size".to_string()
            )
        );
    }
}
