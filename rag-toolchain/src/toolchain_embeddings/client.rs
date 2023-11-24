use serde::{Deserialize, Serialize};

// This can just implement embeddings endpoint
// We want to include exponential backoff and retries for reliability
// also research methods on how not to get rate limited

/// # EmbeddingClient
/// Trait for struct that allows embeddings to be generated
pub trait EmbeddingClient {
    // Used a Vec here in case we want to do batch embeddings like for OpenAI
    fn generate_embeddings(&self, text: Vec<String>) -> Result<Vec<f32>, std::io::Error>;
}

pub struct OpenAIClient;

impl OpenAIClient {
    pub fn new() -> Self {
        return OpenAIClient;
    }
}

impl EmbeddingClient for OpenAIClient {
    fn generate_embeddings(&self, text: Vec<String>) -> Result<Vec<f32>, std::io::Error> {
        Ok(vec![0.0])
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct BatchEmbeddingRequest {
    pub input: Vec<String>,
    pub model: OpenAIEmbeddingModel,
    #[serde(rename = "encoding_format", skip_serializing_if = "Option::is_none")]
    pub encoding_format: Option<EncodingFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub struct EmbeddingRequest {
    pub input: String,
    pub model: OpenAIEmbeddingModel,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encoding_format: Option<EncodingFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub struct EmbeddingResponse {
    pub object: String,
    pub data: Vec<EmbeddingObject>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub struct EmbeddingObject {
    pub object: String,
    pub embedding: Vec<f32>,
    pub index: usize,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub struct Usage {
    pub prompt_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OpenAIEmbeddingModel {
    #[serde(rename = "text-embedding-ada-002")]
    TextEmbeddingAda002,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EncodingFormat {
    Float,
    Base64
}

#[cfg(test)]
mod tests {
    use super::*;

    const EMBEDDING_REQUEST: &'static str = r#"{"input":"Your text string goes here","model":"text-embedding-ada-002"}"#; 
    const EMBEDDING_REQUEST_WITH_OPTIONAL_FIELDS: &'static str = r#"{"input":"Your text string goes here","model":"text-embedding-ada-002","encoding_format":"float","user":"some_user"}"#; 
    const BATCH_EMBEDDING_REQUEST: &'static str = r#"{"input":["Your text string goes here","Second item"],"model":"text-embedding-ada-002"}"#; 

    #[test]
    fn test_embedding_request_without_optional_fields_deserializes() {
        let embedding_request: EmbeddingRequest = serde_json::from_str(EMBEDDING_REQUEST).unwrap();
        assert_eq!(embedding_request.input, "Your text string goes here".to_string());
        assert_eq!(embedding_request.model, OpenAIEmbeddingModel::TextEmbeddingAda002);
        assert_eq!(embedding_request.encoding_format, None);
    }

    #[test]
    fn test_embedding_request_without_optional_fields_serializes() {
        let embedding_request: EmbeddingRequest = EmbeddingRequest {  
            input: "Your text string goes here".to_string(),
            model: OpenAIEmbeddingModel::TextEmbeddingAda002,
            encoding_format: None,
            user: None,
        };

        let serialized_embedding_request = serde_json::to_string(&embedding_request).unwrap();
        assert_eq!(serialized_embedding_request, EMBEDDING_REQUEST);
    }

    #[test]
    fn test_embedding_request_with_optional_fields_deserializes() {
        let embedding_request: EmbeddingRequest = serde_json::from_str(EMBEDDING_REQUEST_WITH_OPTIONAL_FIELDS).unwrap();
        assert_eq!(embedding_request.input, "Your text string goes here".to_string());
        assert_eq!(embedding_request.model, OpenAIEmbeddingModel::TextEmbeddingAda002);
        assert_eq!(embedding_request.encoding_format, EncodingFormat::Float.into());
        assert_eq!(embedding_request.user, "some_user".to_string().into());
    }

    #[test]
    fn test_embedding_request_with_optional_fields_serializes() {
        let embedding_request: EmbeddingRequest = EmbeddingRequest {
            input: "Your text string goes here".to_string(),
            model: OpenAIEmbeddingModel::TextEmbeddingAda002,
            encoding_format: Some(EncodingFormat::Float),
            user: Some("some_user".to_string()),
        };

        let serialized_embedding_request = serde_json::to_string(&embedding_request).unwrap();
        assert_eq!(serialized_embedding_request, EMBEDDING_REQUEST_WITH_OPTIONAL_FIELDS);
        println!("Serialized JSON: {}", serialized_embedding_request);
    }

    #[test]
    fn test_batch_embedding_request_deserializes() {
        let batch_embedding_request: BatchEmbeddingRequest = serde_json::from_str(BATCH_EMBEDDING_REQUEST).unwrap();
        assert_eq!(batch_embedding_request.input.len(), 2);
        assert_eq!(batch_embedding_request.input[0], "Your text string goes here".to_string());
        assert_eq!(batch_embedding_request.input[1], "Second item".to_string());
        assert_eq!(batch_embedding_request.model, OpenAIEmbeddingModel::TextEmbeddingAda002);
    }

    #[test]
    fn test_batch_embedding_request_serializes() {
        let batch_embedding_request: BatchEmbeddingRequest = BatchEmbeddingRequest {
            input: vec![
                "Your text string goes here".to_string(),
                "Second item".to_string(),
            ],
            model: OpenAIEmbeddingModel::TextEmbeddingAda002,
            encoding_format: None, 
            user: None,

        };

        let serialized_batch_embedding_request = serde_json::to_string(&batch_embedding_request).unwrap();
        assert_eq!(serialized_batch_embedding_request, BATCH_EMBEDDING_REQUEST);
        println!("Serialized JSON: {}", serialized_batch_embedding_request);
    }
}
 
