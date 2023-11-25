use dotenv::dotenv;
use reqwest::blocking::Client;
use reqwest::blocking::Response;
use reqwest::header::{HeaderValue, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use std::env;
use std::env::VarError;
use typed_builder::TypedBuilder;

const OPENAI_EMBEDDING_URL: &'static str = "https://api.openai.com/v1/embeddings";

/// # OpenAIEmbeddingClient
/// Allows for interacting with the OpenAI API to generate embeddings
/// Can either embed a single string or a batch of strings
///
/// # Required Environment Variables
/// OPENAI_API_KEY: The API key to use for the OpenAI API
pub trait OpenAIEmbeddingClient {
    // Used a Vec here in case we want to do batch embeddings like for OpenAI
    fn generate_embeddings(
        &self,
        text: Vec<String>,
    ) -> Result<Vec<(String, Vec<f32>)>, std::io::Error>;
}

pub struct OpenAIClient {
    api_key: String,
    client: Client,
}

impl OpenAIClient {
    pub fn new() -> Result<OpenAIClient, VarError> {
        dotenv().ok();
        let api_key: String = match env::var::<String>("OPENAI_API_KEY".into()) {
            Ok(api_key) => api_key,
            Err(e) => return Err(e),
        };
        let client = Client::new();

        Ok(OpenAIClient { api_key, client })
    }

    fn build_request(&self, text: Vec<String>) -> reqwest::blocking::RequestBuilder {
        let request_body = BatchEmbeddingRequest::builder()
            .input(text)
            .model(OpenAIEmbeddingModel::TextEmbeddingAda002)
            .build();
        let content_type = HeaderValue::from_static("application/json");
        let request = self
            .client
            .post(OPENAI_EMBEDDING_URL)
            .bearer_auth(self.api_key.clone())
            .header(CONTENT_TYPE, content_type)
            .json(&request_body);

        return request;
    }

    fn handle_error_response() -> std::io::Error {
        // Map response objects into some form of enum error
        return std::io::Error::new(std::io::ErrorKind::Other, "Error");
    }

    fn handle_success_response(response: Response) -> Vec<(String, Vec<f32>)> {
        // Map response objects into string embedding pairs
        return vec![(String::from("test"), vec![1.0; 1536])];
    }
}

impl OpenAIEmbeddingClient for OpenAIClient {
    fn generate_embeddings(
        &self,
        text: Vec<String>,
    ) -> Result<Vec<(String, Vec<f32>)>, std::io::Error> {
        // Build the request to send to OpenAI
        let request = self.build_request(text);
        // Send the request to OpenAI
        let response: Response = match request.send() {
            Ok(response) => response,
            Err(e) => return Err(std::io::Error::new(std::io::ErrorKind::Other, e)),
        };

        match response.status().is_success() {
            true => return Ok(OpenAIClient::handle_success_response(response)),
            false => return Err(OpenAIClient::handle_error_response()),
        };
    }
}

// Add docs for all the below

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, TypedBuilder)]
#[serde(rename_all = "snake_case")]
pub struct BatchEmbeddingRequest {
    pub input: Vec<String>,
    pub model: OpenAIEmbeddingModel,
    #[serde(rename = "encoding_format", skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub encoding_format: Option<EncodingFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub user: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, TypedBuilder)]
#[serde(rename_all = "snake_case")]
pub struct EmbeddingRequest {
    pub input: String,
    pub model: OpenAIEmbeddingModel,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub encoding_format: Option<EncodingFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub user: Option<String>,
}

#[derive(Debug, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub struct EmbeddingResponse {
    pub data: Vec<EmbeddingObject>,
    pub model: String,
    pub object: String,
    pub usage: Usage,
}

#[derive(Debug, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub struct EmbeddingObject {
    pub embedding: Vec<f32>,
    pub index: usize,
    pub object: String,
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
    Base64,
}

#[cfg(test)]
mod request_model_tests {

    // Tests for the EmbeddingRequest and BatchEmbeddingRequest, as well as the EmbeddingResponse models

    use super::*;

    const EMBEDDING_REQUEST: &'static str =
        r#"{"input":"Your text string goes here","model":"text-embedding-ada-002"}"#;
    const EMBEDDING_REQUEST_WITH_OPTIONAL_FIELDS: &'static str = r#"{"input":"Your text string goes here","model":"text-embedding-ada-002","encoding_format":"float","user":"some_user"}"#;
    const BATCH_EMBEDDING_REQUEST: &'static str = r#"{"input":["Your text string goes here","Second item"],"model":"text-embedding-ada-002"}"#;

    #[test]
    fn test_embedding_request_without_optional_fields_deserializes() {
        let embedding_request: EmbeddingRequest = serde_json::from_str(EMBEDDING_REQUEST).unwrap();
        assert_eq!(
            embedding_request.input,
            "Your text string goes here".to_string()
        );
        assert_eq!(
            embedding_request.model,
            OpenAIEmbeddingModel::TextEmbeddingAda002
        );
        assert_eq!(embedding_request.encoding_format, None);
    }

    #[test]
    fn test_embedding_request_without_optional_fields_serializes() {
        let embedding_request: EmbeddingRequest = EmbeddingRequest::builder()
            .input("Your text string goes here".to_string())
            .model(OpenAIEmbeddingModel::TextEmbeddingAda002)
            .build();

        let serialized_embedding_request = serde_json::to_string(&embedding_request).unwrap();
        assert_eq!(serialized_embedding_request, EMBEDDING_REQUEST);
    }

    #[test]
    fn test_embedding_request_with_optional_fields_deserializes() {
        let embedding_request: EmbeddingRequest =
            serde_json::from_str(EMBEDDING_REQUEST_WITH_OPTIONAL_FIELDS).unwrap();
        assert_eq!(
            embedding_request.input,
            "Your text string goes here".to_string()
        );
        assert_eq!(
            embedding_request.model,
            OpenAIEmbeddingModel::TextEmbeddingAda002
        );
        assert_eq!(
            embedding_request.encoding_format,
            EncodingFormat::Float.into()
        );
        assert_eq!(embedding_request.user, "some_user".to_string().into());
    }

    #[test]
    fn test_embedding_request_with_optional_fields_serializes() {
        let embedding_request: EmbeddingRequest = EmbeddingRequest::builder()
            .input("Your text string goes here".to_string())
            .model(OpenAIEmbeddingModel::TextEmbeddingAda002)
            .encoding_format(EncodingFormat::Float)
            .user("some_user".to_string())
            .build();

        let serialized_embedding_request = serde_json::to_string(&embedding_request).unwrap();
        assert_eq!(
            serialized_embedding_request,
            EMBEDDING_REQUEST_WITH_OPTIONAL_FIELDS
        );
        println!("Serialized JSON: {}", serialized_embedding_request);
    }

    #[test]
    fn test_batch_embedding_request_deserializes() {
        let batch_embedding_request: BatchEmbeddingRequest =
            serde_json::from_str(BATCH_EMBEDDING_REQUEST).unwrap();
        assert_eq!(batch_embedding_request.input.len(), 2);
        assert_eq!(
            batch_embedding_request.input[0],
            "Your text string goes here".to_string()
        );
        assert_eq!(batch_embedding_request.input[1], "Second item".to_string());
        assert_eq!(
            batch_embedding_request.model,
            OpenAIEmbeddingModel::TextEmbeddingAda002
        );
    }

    #[test]
    fn test_batch_embedding_request_serializes() {
        let batch_embedding_request: BatchEmbeddingRequest = BatchEmbeddingRequest::builder()
            .input(vec![
                "Your text string goes here".to_string(),
                "Second item".to_string(),
            ])
            .model(OpenAIEmbeddingModel::TextEmbeddingAda002)
            .build();

        let serialized_batch_embedding_request =
            serde_json::to_string(&batch_embedding_request).unwrap();
        assert_eq!(serialized_batch_embedding_request, BATCH_EMBEDDING_REQUEST);
        println!("Serialized JSON: {}", serialized_batch_embedding_request);
    }

    const EMBEDDING_RESPONSE: &'static str = r#"{"data":[{"embedding":[-0.006929283495992422,-0.005336422007530928,-0.009327292,-0.024047505110502243],"index":0,"object":"embedding"}],"model":"text-embedding-ada-002","object":"list","usage":{"prompt_tokens":5,"total_tokens":5}}"#;

    #[test]
    fn test_embedding_response_deserializes() {
        let embedding_response: EmbeddingResponse =
            serde_json::from_str(EMBEDDING_RESPONSE).unwrap();
        assert_eq!(embedding_response.data.len(), 1);
        assert_eq!(embedding_response.data[0].embedding.len(), 4);
        assert_eq!(embedding_response.data[0].index, 0);
        assert_eq!(embedding_response.data[0].object, "embedding");
        assert_eq!(
            embedding_response.model,
            "text-embedding-ada-002".to_string()
        );
        assert_eq!(embedding_response.object, "list");
        assert_eq!(embedding_response.usage.prompt_tokens, 5);
        assert_eq!(embedding_response.usage.total_tokens, 5);
    }

    #[test]
    fn test_request() {
        let client = OpenAIClient::new().unwrap();
        let result = client.generate_embeddings(vec![
            "This is a test string".to_string(),
            "This is another test string".to_string(),
        ]);
        println!("{:?}", result);
    }
}
