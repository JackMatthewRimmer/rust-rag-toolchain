/// # Clients
/// This module will contain all of the client code for different services
/// that can be used to interact with Gen AI models.
#[cfg(feature = "openai")]
mod open_ai;

#[cfg(feature = "anthropic")]
mod anthropic;

mod traits;
mod types;

#[cfg(feature = "openai")]
pub use self::open_ai::{
    CompletionStreamValue, OpenAIChatCompletionClient, OpenAICompletionStream,
    OpenAIEmbeddingClient, OpenAIError, OpenAIModel,
};

#[cfg(feature = "anthropic")]
pub use self::anthropic::{AnthropicChatCompletionClient, AnthropicError, AnthropicModel};

pub use self::traits::{
    AsyncChatClient, AsyncEmbeddingClient, AsyncStreamedChatClient, ChatCompletionStream,
};
pub use self::types::PromptMessage;

// Export the trait mocks for use in testing
#[cfg(test)]
pub use traits::{MockAsyncChatClient, MockAsyncStreamedChatClient, MockChatCompletionStream};
