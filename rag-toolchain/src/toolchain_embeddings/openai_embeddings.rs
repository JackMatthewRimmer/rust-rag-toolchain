use crate::toolchain_embeddings::embedding_models::AsyncEmbeddingClient;
use crate::toolchain_indexing::types::{Chunk, Chunks, Embedding};
use async_trait::async_trait;
use dotenv::dotenv;
use reqwest::header::{HeaderValue, CONTENT_TYPE};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::env;
use std::env::VarError;
use std::fmt::Display;
use typed_builder::TypedBuilder;

const OPENAI_EMBEDDING_URL: &str = "https://api.openai.com/v1/embeddings";

/// # OpenAIEmbeddingClient
/// Allows for interacting with the OpenAI API to generate embeddings
/// Can either embed a single string or a batch of strings
///
/// # Required Environment Variables
/// OPENAI_API_KEY: The API key to use for the OpenAI API

pub struct OpenAIClient {
    api_key: String,
    client: Client,
    url: String, // This was done to support mocking
}

impl OpenAIClient {
    /// Create a new OpenAIClient.
    /// Must have the OPENAI_API_KEY environment variable set
    pub fn new() -> Result<OpenAIClient, VarError> {
        dotenv().ok();
        let api_key: String = match env::var::<String>("OPENAI_API_KEY".into()) {
            Ok(api_key) => api_key,
            Err(e) => return Err(e),
        };
        let client: Client = Client::new();

        Ok(OpenAIClient {
            api_key,
            client,
            url: OPENAI_EMBEDDING_URL.into(),
        })
    }

    /// Sends a request to the OpenAI API and returns the response
    ///
    /// # Arguments
    /// * `request` - The request to send to the OpenAI API
    /// this request should come prebuilt ready to call .send() on
    ///
    /// # Errors
    /// * `OpenAIError::ErrorSendingRequest` - if request.send() errors
    /// * `OpenAIError::ErrorGettingResponseBody` - if response.text() errors
    /// * `OpenAIError::ErrorDeserializingResponseBody` - if serde_json::from_str() errors
    /// * `OpenAIError` - if the response code is not 200 this can be any of the associates status
    ///    code errors or variatn of `OpenAIError::UNDEFINED`
    ///
    /// # Returns
    /// * `EmbeddingResponse` - The deserialized response from OpenAI
    async fn send_embedding_request(
        request: reqwest::RequestBuilder,
    ) -> Result<EmbeddingResponse, OpenAIError> {
        let response: reqwest::Response = match request.send().await {
            Ok(response) => response,
            Err(e) => return Err(OpenAIError::ErrorSendingRequest(e.to_string())),
        };
        if !response.status().is_success() {
            return Err(OpenAIClient::handle_error_response(response).await);
        }
        let response_body: String = response
            .text()
            .await
            .map_err(|error| OpenAIError::ErrorGettingResponseBody(error.to_string()))?;

        let embedding_response: EmbeddingResponse = match serde_json::from_str(&response_body) {
            Err(e) => {
                return Err(OpenAIError::ErrorDeserializingResponseBody(e.to_string()));
            }
            Ok(embedding_response) => embedding_response,
        };
        Ok(embedding_response)
    }

    /// Explicit error mapping between response codes and error types
    ///
    /// # Arguments
    /// `response` - The reqwest response from OpenAI
    ///
    /// # Returns
    /// `OpenAIError` - The error type that maps to the response code
    async fn handle_error_response(response: reqwest::Response) -> OpenAIError {
        // Map response objects into some form of enum error
        let status_code = response.status().as_u16();
        let body_text = match response.text().await {
            Ok(text) => text,
            Err(e) => return OpenAIError::UNDEFINED(status_code, e.to_string()),
        };
        println!("Error Body: {}", body_text);
        let error_body: OpenAIErrorBody = match serde_json::from_str(&body_text) {
            Ok(error_body) => error_body,
            Err(e) => return OpenAIError::UNDEFINED(status_code, e.to_string()),
        };
        match status_code {
            401 => OpenAIError::CODE401(error_body),
            429 => OpenAIError::CODE429(error_body),
            500 => OpenAIError::CODE500(error_body),
            503 => OpenAIError::CODE503(error_body),
            undefined => OpenAIError::UNDEFINED(undefined, body_text),
        }
    }

    /// Takes a successful response and maps it into a vector of string embedding pairs
    ///
    /// # Arguments
    /// `input_text` - The input text that was sent to OpenAI
    /// `response` - The deserialized response from OpenAI
    ///
    /// # Returns
    /// `Vec<(String, Vec<f32>)>` - A vector of string embedding pairs the can be stored
    fn handle_success_response(
        input_text: Chunks,
        response: EmbeddingResponse,
    ) -> Vec<(Chunk, Embedding)> {
        // Map response objects into string embedding pairs
        let embedding_objects: Vec<EmbeddingObject> = response.data;

        let embeddings: Vec<Embedding> =
            Embedding::iter_to_vec(embedding_objects.iter().map(|obj| obj.embedding.clone()));

        let input_text: Vec<Chunk> = input_text.to_vec::<Chunk>();
        input_text.into_iter().zip(embeddings).collect()
    }
}

#[async_trait]
impl AsyncEmbeddingClient for OpenAIClient {
    type ErrorType = OpenAIError;

    async fn generate_embeddings(
        &self,
        text: Chunks,
    ) -> Result<Vec<(Chunk, Embedding)>, OpenAIError> {
        let input_text: Vec<String> = text.to_vec::<String>();
        let request_body = BatchEmbeddingRequest::builder()
            .input(input_text)
            .model(OpenAIEmbeddingModel::TextEmbeddingAda002)
            .build();
        let content_type = HeaderValue::from_static("application/json");
        let request: reqwest::RequestBuilder = self
            .client
            .post(self.url.clone())
            .bearer_auth(self.api_key.clone())
            .header(CONTENT_TYPE, content_type)
            .json(&request_body);

        let embedding_response: EmbeddingResponse =
            OpenAIClient::send_embedding_request(request).await?;
        Ok(OpenAIClient::handle_success_response(
            text.clone(),
            embedding_response,
        ))
    }

    async fn generate_embedding(&self, text: Chunk) -> Result<(Chunk, Embedding), Self::ErrorType> {
        let request_body = EmbeddingRequest::builder()
            .input(text.clone().into())
            .model(OpenAIEmbeddingModel::TextEmbeddingAda002)
            .build();
        let content_type = HeaderValue::from_static("application/json");
        let request: reqwest::RequestBuilder = self
            .client
            .post(self.url.clone())
            .bearer_auth(self.api_key.clone())
            .header(CONTENT_TYPE, content_type)
            .json(&request_body);
        let embedding_response: EmbeddingResponse =
            OpenAIClient::send_embedding_request(request).await?;
        Ok(
            OpenAIClient::handle_success_response(vec![text.clone()].into(), embedding_response)[0]
                .clone(),
        )
    }
}

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
    // For some reason this comes back as text-embedding-ada-002-v2
    #[serde(rename = "text-embedding-ada-002")]
    TextEmbeddingAda002,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EncodingFormat {
    Float,
    Base64,
}

#[derive(Debug, Deserialize, PartialEq)]
pub struct OpenAIErrorBody {
    pub error: OpenAIErrorData,
}

#[derive(Debug, Deserialize, PartialEq)]
pub struct OpenAIErrorData {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    pub param: Option<String>,
    pub code: String,
}

#[derive(Debug, PartialEq)]
pub enum OpenAIError {
    /// # Invalid Authentication or Incorrect API Key provided
    CODE401(OpenAIErrorBody),
    /// # Rate limit reached or Monthly quota exceeded
    CODE429(OpenAIErrorBody),
    /// # Server Error
    CODE500(OpenAIErrorBody),
    /// # The engine is currently overloaded
    CODE503(OpenAIErrorBody),
    /// # Missed cases for error codes, includes Status Code and Error Body as a string
    UNDEFINED(u16, String),
    ErrorSendingRequest(String),
    ErrorGettingResponseBody(String),
    ErrorDeserializingResponseBody(String),
}

impl std::error::Error for OpenAIError {}
impl Display for OpenAIError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OpenAIError::CODE401(error_body) => write!(
                f,
                "Invalid Authentication or Incorrect API Key provided: {}",
                error_body.error.message
            ),
            OpenAIError::CODE429(error_body) => write!(
                f,
                "Rate limit reached or Monthly quota exceeded: {}",
                error_body.error.message
            ),
            OpenAIError::CODE500(error_body) => {
                write!(f, "Server Error: {}", error_body.error.message)
            }
            OpenAIError::CODE503(error_body) => write!(
                f,
                "The engine is currently overloaded: {}",
                error_body.error.message
            ),
            OpenAIError::UNDEFINED(status_code, error_body) => {
                write!(f, "Undefined Error. This should not happen, if this is a missed error please report it: https://github.com/JackMatthewRimmer/rust-rag-toolchain: {} - {}", status_code, error_body)
            }
            OpenAIError::ErrorSendingRequest(error) => {
                write!(f, "Error Sending Request: {}", error)
            }
            OpenAIError::ErrorGettingResponseBody(error) => {
                write!(f, "Error Getting Response Body: {}", error)
            }
            OpenAIError::ErrorDeserializingResponseBody(error) => {
                write!(f, "Error Deserializing Response Body: {}", error)
            }
        }
    }
}

#[cfg(test)]
mod client_tests {
    use crate::toolchain_embeddings::embedding_models::AsyncEmbeddingClient;
    use crate::toolchain_embeddings::openai_embeddings::OpenAIClient;
    use crate::toolchain_indexing::types::{Chunk, Chunks, Embedding};
    const EMBEDDING_RESPONSE: &'static str = r#"
    {
        "data": [
            {
                "embedding": [
                    -0.006929283495992422,
                    -0.005336422007530928,
                    -0.009327292,
                    -0.024047505110502243
                ],
                "index": 0,
                "object": "embedding"
            },
            {
                "embedding": [
                    -0.006929283495992422,
                    -0.005336422007530928,
                    -0.009327292,
                    -0.024047505110502243
                ],
                "index": 1,
                "object": "embedding"
            }
        ],
        "model": "text-embedding-ada-002",
        "object": "list",
        "usage": {
            "prompt_tokens": 5,
            "total_tokens": 5
        }
    }
    "#;

    #[tokio::test]
    async fn test_correct_response_succeeds() {
        std::env::set_var("OPENAI_API_KEY", "fake key");
        let mut server = mockito::Server::new();
        let url = server.url();

        let mut client: OpenAIClient = OpenAIClient::new().unwrap();
        client.url = url.clone();

        let mock = server
            .mock("POST", "/")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(EMBEDDING_RESPONSE)
            .create();

        let chunks: Chunks = Chunks::from(vec![Chunk::from("Test-0"), Chunk::from("Test-1")]);
        let response = client.generate_embeddings(chunks).await.unwrap();
        mock.assert();
        for (i, (chunk, embedding)) in response.into_iter().enumerate() {
            assert_eq!(chunk, Chunk::from(format!("Test-{}", i)));
            assert_eq!(
                embedding,
                Embedding::from(vec![
                    -0.006929283495992422,
                    -0.005336422007530928,
                    -0.009327292,
                    -0.024047505110502243
                ])
            );
        }
    }
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
}
