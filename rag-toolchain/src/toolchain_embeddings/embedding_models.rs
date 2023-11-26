use tiktoken_rs::tokenizer::Tokenizer;

/// This enum is used to hold all embedding models and their metadata
pub enum EmbeddingModels {
    OpenAI(OpenAIEmbeddingModel),
}

// Enum to hold all OpenAI embedding models and there metadata
pub enum OpenAIEmbeddingModel {
    TextEmbeddingAda002,
}

/// This method returns the static metadata for the embedding model
impl OpenAIEmbeddingModel {
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

/// This method converts the EmbeddingModels enum into the OpenAIEmbeddingModel enum
impl From<OpenAIEmbeddingModel> for EmbeddingModels {
    fn from(model: OpenAIEmbeddingModel) -> Self {
        EmbeddingModels::OpenAI(model)
    }
}

/// This method converts the EmbeddingModels enum into the OpenAIEmbeddingModel enum
impl From<EmbeddingModels> for OpenAIEmbeddingModel {
    fn from(model: EmbeddingModels) -> Self {
        match model {
            EmbeddingModels::OpenAI(model) => model,
        }
    }
}

/// Struct to contain static metadata for OpenAI embedding models
pub struct OpenAIEmbeddingMetadata {
    pub dimensions: usize,
    pub max_tokens: usize,
    pub tokenizer: Tokenizer,
}
