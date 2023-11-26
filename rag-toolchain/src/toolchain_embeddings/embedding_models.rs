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

impl From<OpenAIEmbeddingModel> for EmbeddingModels {
    fn from(model: OpenAIEmbeddingModel) -> Self {
        EmbeddingModels::OpenAI(model)
    }
}

impl From<EmbeddingModels> for OpenAIEmbeddingModel {
    fn from(model: EmbeddingModels) -> Self {
        match model {
            EmbeddingModels::OpenAI(model) => model,
        }
    }
}

pub struct OpenAIEmbeddingMetadata {
    pub dimensions: usize,
    pub max_tokens: usize,
    pub tokenizer: Tokenizer,
}
