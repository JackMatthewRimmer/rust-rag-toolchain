use crate::common::types::{Chunk, Chunks, Embedding};
use async_trait::async_trait;
use std::error::Error;

use super::types::PromptMessage;

/// # AsyncEmbeddingClient
/// Trait for any client that generates embeddings asynchronously
#[async_trait]
pub trait AsyncEmbeddingClient {
    type ErrorType: Error;
    async fn generate_embedding(&self, text: Chunk) -> Result<(Chunk, Embedding), Self::ErrorType>;
    async fn generate_embeddings(
        &self,
        text: Chunks,
    ) -> Result<Vec<(Chunk, Embedding)>, Self::ErrorType>;
}

/// # AsyncEmbeddingClient
/// Trait for any client that generates chat completions asynchronously

// TODO: multiple options here to allow for using an N parameter and such
#[async_trait]
pub trait AsyncChatClient {
    type ErrorType: Error;
    async fn invoke(
        &self,
        prompt_messages: Vec<PromptMessage>,
    ) -> Result<PromptMessage, Self::ErrorType>;
}
