use tiktoken_rs::tokenizer::Tokenizer;

// Enum to hold all embedding models
pub enum EmbeddingModels {
    OpenAI(OpenAIEmbeddingModel),
}

// Enum to hold all OpenAI embedding models and there metadata
pub enum OpenAIEmbeddingModel {
    TextEmbeddingAda002,
}

impl OpenAIEmbeddingModel {
    pub fn new(model: OpenAIEmbeddingModel) -> EmbeddingModels {
        EmbeddingModels::OpenAI(model)
    }

    pub fn metadata(&self) -> OpenAIEmbeddingMetadata {
        match self {
            OpenAIEmbeddingModel::TextEmbeddingAda002 => OpenAIEmbeddingMetadata {
                dimensions: 1536,
                max_tokens: 8192,
                tokenizer: Tokenizer::Cl100kBase,
            },
        }
    }
}
pub struct OpenAIEmbeddingMetadata {
    pub dimensions: usize,
    pub max_tokens: usize,
    pub tokenizer: Tokenizer,
}
