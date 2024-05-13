#[cfg(feature = "openai")]
mod model;
#[cfg(feature = "openai")]
pub mod open_ai_chat_completions;
#[cfg(feature = "openai")]
mod open_ai_core;
#[cfg(feature = "openai")]
pub mod open_ai_embeddings;

#[cfg(feature = "openai")]
pub use self::model::chat_completions::OpenAIModel;
#[cfg(feature = "openai")]
pub use self::model::errors::OpenAIError;
