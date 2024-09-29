#[cfg(feature = "openai")]
mod model;
#[cfg(feature = "openai")]
mod open_ai_chat_completions;
#[cfg(feature = "openai")]
mod open_ai_core;
#[cfg(feature = "openai")]
mod open_ai_embeddings;

#[cfg(feature = "openai")]
pub use self::model::{chat_completions::OpenAIModel, errors::OpenAIError};

#[cfg(feature = "openai")]
pub use self::open_ai_chat_completions::{
    CompletionStreamValue, OpenAIChatCompletionClient, OpenAICompletionStream,
};

#[cfg(feature = "openai")]
pub use self::open_ai_embeddings::OpenAIEmbeddingClient;
