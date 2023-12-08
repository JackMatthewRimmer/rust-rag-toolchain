use crate::toolchain_indexing::types::{Chunk, Chunks, Embedding};
use async_trait::async_trait;
use std::error::Error;
use tiktoken_rs::tokenizer::Tokenizer;
use tiktoken_rs::CoreBPE;

/// # AsyncEmbeddingClient
/// Trait for any client that generates embeddings asynchronously
/// such as OpenAI
#[async_trait]
pub trait AsyncEmbeddingClient {
    type ErrorType: Error;
    async fn generate_embedding(&self, text: Chunk) -> Result<(Chunk, Embedding), Self::ErrorType>;
    async fn generate_embeddings(
        &self,
        text: Chunks,
    ) -> Result<Vec<(Chunk, Embedding)>, Self::ErrorType>;
}

/// # EmbeddingClient
/// Trait for any client that generates embeddings synchronously
/// can be used for any local embedding model
pub trait EmbeddingClient {
    type ErrorType: Error;
    fn generate_embedding(&self, text: Chunk) -> Result<(Chunk, Embedding), Self::ErrorType>;
    fn generate_embeddings(&self, text: Chunks)
        -> Result<Vec<(Chunk, Embedding)>, Self::ErrorType>;
}

// ---------------------- Embedding Models ----------------------
/// # HasMetadata
/// This trait is used for methods to understand the requirements
/// set out by which embedding model is being used such as embedding
/// dimensions and max tokens
pub trait HasMetadata {
    fn metadata(&self) -> EmbeddingModelMetadata;
}

/// # EmbeddingModelMetadata
/// Struct to contain all of the relevant metadata for an embedding model
pub struct EmbeddingModelMetadata {
    pub dimensions: usize,
    pub max_tokens: usize,
    pub tokenizer: Box<dyn TokenizerWrapper>,
}

/// # TokenizerWrapper
/// We wrap the tokenizer for a specific embedding model to allow
/// for a common interface for tokenization
pub trait TokenizerWrapper {
    // This should potentially go back to a Result
    fn tokenize(&self, text: &str) -> Option<Vec<String>>;
}
// -------------------------------------------------------------

// ------------------ OpenAI Embedding Models ------------------
/// # OpenAIEmbeddingModel
/// Top level enum to hold all OpenAI embedding model variants
pub enum OpenAIEmbeddingModel {
    TextEmbeddingAda002,
}

// Match the embedding model to the metadata
impl HasMetadata for OpenAIEmbeddingModel {
    fn metadata(&self) -> EmbeddingModelMetadata {
        match self {
            OpenAIEmbeddingModel::TextEmbeddingAda002 => EmbeddingModelMetadata {
                dimensions: 1536,
                max_tokens: 8192,
                tokenizer: Box::new(OpenAITokenizer::new(Tokenizer::Cl100kBase)),
            },
        }
    }
}

// Wrap tiktoken_rs tokenizer for OpenAI
struct OpenAITokenizer {
    bpe: CoreBPE,
}

// Added new function to hide the unwrap
// The panic here should be fine as this shouldn't fail as we use an enum variant
impl OpenAITokenizer {
    pub fn new(model: Tokenizer) -> Self {
        OpenAITokenizer {
            bpe: tiktoken_rs::get_bpe_from_tokenizer(model).unwrap(),
        }
    }
}

// Implement the tokenize function for OpenAITokenizer
impl TokenizerWrapper for OpenAITokenizer {
    fn tokenize(&self, text: &str) -> Option<Vec<String>> {
        if let Ok(tokens) = self.bpe.split_by_token(text, true) {
            Some(tokens)
        } else {
            None
        }
    }
}
// ------------------ OpenAI Embedding Models ------------------
