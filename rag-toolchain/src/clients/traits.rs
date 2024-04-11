use crate::common::{Chunk, Chunks, Embedding};
use std::error::Error;
use std::future::Future;

use super::types::PromptMessage;

/// # [`AsyncEmbeddingClient`]
/// Trait for any client that generates embeddings asynchronously
pub trait AsyncEmbeddingClient {
    type ErrorType: Error;
    fn generate_embedding(
        &self,
        text: Chunk,
    ) -> impl Future<Output = Result<Embedding, Self::ErrorType>> + Send;
    fn generate_embeddings(
        &self,
        text: Chunks,
    ) -> impl Future<Output = Result<Vec<Embedding>, Self::ErrorType>> + Send;
}

/// # [`AsyncChatClient`]
/// Trait for any client that generates chat completions asynchronously
pub trait AsyncChatClient {
    type ErrorType: Error;
    fn invoke(
        &self,
        prompt_messages: Vec<PromptMessage>,
    ) -> impl Future<Output = Result<PromptMessage, Self::ErrorType>> + Send;
}

pub trait AsyncStreamedChatClient {
    type ErrorType: Error;
    type Item: ChatCompletionStream;
    fn invoke_stream(
        &self,
        prompt_messages: Vec<PromptMessage>,
    ) -> impl Future<Output = Result<Self::Item, Self::ErrorType>> + Send;
}

/// # [`ChatCompletionStream`]
///
/// Trait for any stream that generates chat completions
pub trait ChatCompletionStream {
    type ErrorType: Error;
    type Item;
    fn next(&mut self) -> impl Future<Output = Option<Result<Self::Item, Self::ErrorType>>>;
}

#[cfg(test)]
use mockall::*;
#[cfg(test)]
mock! {
    pub AsyncEmbeddingClient {}
    impl AsyncEmbeddingClient for AsyncEmbeddingClient {
        type ErrorType = std::io::Error;
        async fn generate_embedding(&self, text: Chunk) -> Result<Embedding, <Self as AsyncEmbeddingClient>::ErrorType>;
        async fn generate_embeddings(
            &self,
            text: Chunks,
        ) -> Result<Vec<Embedding>, <Self as AsyncEmbeddingClient>::ErrorType>;
    }
}
#[cfg(test)]
mock! {
    pub AsyncChatClient {}
    impl AsyncChatClient for AsyncChatClient {
        type ErrorType = std::io::Error;
        async fn invoke(
            &self,
            prompt_messages: Vec<PromptMessage>,
        ) -> Result<PromptMessage, <Self as AsyncChatClient>::ErrorType>;
    }
}

#[cfg(test)]
impl Clone for MockAsyncChatClient {
    fn clone(&self) -> Self {
        MockAsyncChatClient::new()
    }
}
