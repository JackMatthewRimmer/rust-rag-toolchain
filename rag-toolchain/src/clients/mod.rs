/// # Clients
/// This module will contain all of the client code for different services
/// that can be used to interact with Gen AI models.
mod open_ai;
mod traits;
mod types;
pub use self::open_ai::open_ai_chat_completions::{
    CompletionStreamValue, OpenAIChatCompletionClient, OpenAICompletionStream,
};
pub use self::open_ai::open_ai_embeddings::OpenAIEmbeddingClient;
pub use self::open_ai::{OpenAIError, OpenAIModel};
pub use self::traits::{
    AsyncChatClient, AsyncEmbeddingClient, AsyncStreamedChatClient, ChatCompletionStream,
};
pub use self::types::PromptMessage;

// Export the trait mocks for use in testing
#[cfg(test)]
pub use traits::{MockAsyncChatClient, MockAsyncStreamedChatClient, MockChatCompletionStream};
