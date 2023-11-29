use tiktoken_rs::tokenizer::Tokenizer;

/// This enum is used to hold all embedding models and their metadata
pub enum EmbeddingModel {
    OpenAI(OpenAIEmbeddingModel),
}

impl EmbeddingModel {
    pub fn dimensions(&self) -> usize {
        match self {
            EmbeddingModel::OpenAI(model) => model.metadata().dimensions,
        }
    }
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
impl From<OpenAIEmbeddingModel> for EmbeddingModel {
    fn from(model: OpenAIEmbeddingModel) -> Self {
        EmbeddingModel::OpenAI(model)
    }
}

/// This method converts the EmbeddingModels enum into the OpenAIEmbeddingModel enum
impl From<EmbeddingModel> for OpenAIEmbeddingModel {
    fn from(model: EmbeddingModel) -> Self {
        match model {
            EmbeddingModel::OpenAI(model) => model,
        }
    }
}

/// Struct to contain static metadata for OpenAI embedding models
pub struct OpenAIEmbeddingMetadata {
    pub dimensions: usize,
    pub max_tokens: usize,
    pub tokenizer: Tokenizer,
}
