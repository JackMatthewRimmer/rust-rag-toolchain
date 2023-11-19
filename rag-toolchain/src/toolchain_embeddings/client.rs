use serde::{Deserialize, Serialize};

// This can just implement embeddings endpoint
// We want to include exponential backoff and retries for reliability
// also research methods on how not to get rate limited

pub struct OpenAIClient;

impl OpenAIClient {
    pub fn new() -> Self {
        return OpenAIClient;
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    object: String,
    data: Vec<EmbeddingObject>,
    model: String,
    usage: Usage,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EmbeddingObject {
    object: String,
    embedding: Vec<f32>,
    index: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Usage {
    prompt_tokens: usize,
    total_tokens: usize,
}
