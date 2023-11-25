use tiktoken_rs::{cl100k_base, CoreBPE};

// Enum to hold all embedding models
pub enum EmbeddingModels {
    OpenAI(OpenAIEmbeddingModel),
}

// Enum to hold all OpenAI embedding models and there metadata
pub enum OpenAIEmbeddingModel {
    TextEmbeddingAda002(TextEmbeddingAda002MetaData),
}

pub struct TextEmbeddingAda002MetaData {}
impl TextEmbeddingAda002MetaData {
    pub const DIMENSIONS: usize = 1536;
    pub const MAX_TOKENS: usize = 8192;
    pub const TOKENIZER: &'static str = "cl100k_base";
}

// --------------------------------------------------
// Utility functions
// --------------------------------------------------

// A way to shorthand the meta data for open ai embedding models
pub fn openai_embedding_metadata(model: &OpenAIEmbeddingModel) -> (usize, usize, &str) {
    match model {
        OpenAIEmbeddingModel::TextEmbeddingAda002(_) => {
            return (
                TextEmbeddingAda002MetaData::DIMENSIONS,
                TextEmbeddingAda002MetaData::MAX_TOKENS,
                TextEmbeddingAda002MetaData::TOKENIZER,
            )
        }
    }
}
