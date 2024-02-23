mod open_ai;
mod traits;
mod types;
pub use self::open_ai::open_ai_chat_completions::OpenAIChatCompletionClient;
pub use self::open_ai::open_ai_embeddings::OpenAIEmbeddingClient;
pub use self::open_ai::{OpenAIError, OpenAIModel};
pub use self::traits::{AsyncChatClient, AsyncEmbeddingClient};
pub use self::types::PromptMessage;

// Export the trait mocks for use in testing
#[cfg(test)]
pub use traits::{MockAsyncChatClient, MockAsyncEmbeddingClient};
