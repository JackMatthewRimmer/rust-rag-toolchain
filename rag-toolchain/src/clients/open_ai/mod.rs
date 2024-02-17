mod model;
pub mod open_ai_chat_completions;
mod open_ai_core;
pub mod open_ai_embeddings;

pub use self::model::chat_completions::OpenAIModel;
pub use self::model::errors::OpenAIError;