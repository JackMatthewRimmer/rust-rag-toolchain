use crate::clients::model::embeddings::{
    BatchEmbeddingRequest, EmbeddingObject, EmbeddingRequest, EmbeddingResponse,
};
use crate::clients::model::chat_completions::{ChatCompletionRequest, OpenAIModel};
use crate::clients::model::errors::{OpenAIError, OpenAIErrorBody};
use crate::clients::traits::AsyncEmbeddingClient;
use crate::clients::types::PromptMessage;
use crate::common::embedding_shared::OpenAIEmbeddingModel;
use crate::common::types::{Chunk, Chunks, Embedding};
use async_trait::async_trait;
use dotenv::dotenv;
use reqwest::header::{HeaderValue, CONTENT_TYPE};
use reqwest::{Client, StatusCode};
use std::env;
use std::env::VarError;

use super::model::chat_completions::ChatMessage;
use super::traits::AsyncChatClient;

const OPENAI_EMBEDDING_URL: &str = "https://api.openai.com/v1/embeddings";

/// # OpenAIEmbeddingClient
/// Allows for interacting with the OpenAI API to generate embeddings.
/// You can either embed a single string or a batch of strings.
///
/// # Required Environment Variables
/// OPENAI_API_KEY: The API key to use for the OpenAI API

pub struct OpenAIClient {
    api_key: String,
    client: Client,
    embedding_model: OpenAIEmbeddingModel,
    url: String, // This was done to support mocking
}

impl OpenAIClient {
    /// # try_new
    /// Create a new OpenAIClient.
    /// Must have the OPENAI_API_KEY environment variable set
    ///
    /// # Arguments
    /// * `embedding_model` - The type of embedding model you wish to use.
    /// See <https://platform.openai.com/docs/guides/embeddings/what-are-embeddings>
    ///
    /// # Errors
    /// * [`VarError`] - If the OPENAI_API_KEY environment variable is not set
    ///
    /// # Returns
    /// * [`OpenAIClient`] - The OpenAIClient
    pub fn try_new(embedding_model: OpenAIEmbeddingModel) -> Result<OpenAIClient, VarError> {
        dotenv().ok();
        let api_key: String = match env::var::<String>("OPENAI_API_KEY".into()) {
            Ok(api_key) => api_key,
            Err(e) => return Err(e),
        };
        let client: Client = Client::new();
        let url = OPENAI_EMBEDDING_URL.into();
        Ok(OpenAIClient {
            api_key,
            client,
            embedding_model,
            url,
        })
    }

    // # send_embedding_request
    // Sends a request to the OpenAI API and returns the response
    //
    // # Arguments
    // * `request` - The request to send to the OpenAI API
    // this request should come prebuilt ready to call .send() on
    //
    // # Errors
    // * [`OpenAIError::ErrorSendingRequest`] - if request.send() errors
    // * [`OpenAIError::ErrorGettingResponseBody`] - if response.text() errors
    // * [`OpenAIError::ErrorDeserializingResponseBody`] - if serde_json::from_str() errors
    // * [`OpenAIError`] - if the response code is not 200 this can be any of the associates status
    //    code errors or variatn of `OpenAIError::UNDEFINED`
    //
    // # Returns
    // * `EmbeddingResponse` - The deserialized response from OpenAI
    async fn send_embedding_request(
        request: reqwest::RequestBuilder,
    ) -> Result<EmbeddingResponse, OpenAIError> {
        let response: reqwest::Response = match request.send().await {
            Ok(response) => response,
            Err(e) => return Err(OpenAIError::ErrorSendingRequest(e.to_string())),
        };

        let status_code: StatusCode = response.status();
        if !status_code.is_success() {
            return Err(OpenAIClient::handle_embedding_error_response(response).await);
        }
        let response_body: String = response
            .text()
            .await
            .map_err(|error| OpenAIError::ErrorGettingResponseBody(error.to_string()))?;

        let embedding_response: EmbeddingResponse = match serde_json::from_str(&response_body) {
            Err(e) => {
                return Err(OpenAIError::ErrorDeserializingResponseBody(
                    status_code.as_u16(),
                    e.to_string(),
                ));
            }
            Ok(embedding_response) => embedding_response,
        };
        Ok(embedding_response)
    }

    // # handle_error_response
    // Explicit error mapping between response codes and error types
    //
    // # Arguments
    // `response` - The reqwest response from OpenAI
    //
    // # Returns
    // `OpenAIError` - The error type that maps to the response code
    async fn handle_embedding_error_response(response: reqwest::Response) -> OpenAIError {
        // Map response objects into some form of enum error
        let status_code = response.status().as_u16();
        let body_text = match response.text().await {
            Ok(text) => text,
            Err(e) => return OpenAIError::UNDEFINED(status_code, e.to_string()),
        };
        let error_body: OpenAIErrorBody = match serde_json::from_str(&body_text) {
            Ok(error_body) => error_body,
            Err(e) => {
                return OpenAIError::ErrorDeserializingResponseBody(status_code, e.to_string())
            }
        };
        match status_code {
            400 => OpenAIError::CODE400(error_body),
            401 => OpenAIError::CODE401(error_body),
            429 => OpenAIError::CODE429(error_body),
            500 => OpenAIError::CODE500(error_body),
            503 => OpenAIError::CODE503(error_body),
            undefined => OpenAIError::UNDEFINED(undefined, body_text),
        }
    }

    // # handle_success_response
    // Takes a successful response and maps it into a vector of string embedding pairs
    // assumption made the two iters will zip up 1:1 (as this should be the case)
    //
    // # Arguments
    // `input_text` - The input text that was sent to OpenAI
    // `response` - The deserialized response from OpenAI
    //
    // # Returns
    // `Vec<(String, Vec<f32>)>` - A vector of string embedding pairs the can be stored
    fn handle_embedding_success_response(
        input_text: Chunks,
        response: EmbeddingResponse,
    ) -> Vec<(Chunk, Embedding)> {
        // Map response objects into string embedding pairs
        let embedding_objects: Vec<EmbeddingObject> = response.data;

        let embeddings: Vec<Embedding> =
            Embedding::iter_to_vec(embedding_objects.iter().map(|obj| obj.embedding.clone()));
        input_text.into_iter().zip(embeddings).collect()
    }
}

#[async_trait]
impl AsyncEmbeddingClient for OpenAIClient {
    type ErrorType = OpenAIError;

    /// # generate_embeddings
    /// Function to generate embeddings for [`Chunks`].
    /// Allows you to get an embedding for multiple strings.
    ///
    /// # Arguments
    /// * `text` - The text chunks/strings to generate an embeddings for.
    ///
    /// # Errors
    /// * [`OpenAIError`] - If the request to OpenAI fails.
    ///  
    /// # Returns
    /// * `Result<Vec<(Chunk, Embedding)>, OpenAIError>` - A result containing
    /// pairs of the original text and the embedding that was generated.
    async fn generate_embeddings(
        &self,
        text: Chunks,
    ) -> Result<Vec<(Chunk, Embedding)>, OpenAIError> {
        let input_text: Vec<String> = text
            .iter()
            .map(|chunk| (*chunk).chunk().to_string())
            .collect();

        let request_body = BatchEmbeddingRequest::builder()
            .input(input_text)
            .model(self.embedding_model)
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

        Ok(OpenAIClient::handle_embedding_success_response(
            text,
            embedding_response,
        ))
    }

    /// # generate_embedding
    /// Function to generate an embedding for a [`Chunk`].
    /// Allows you to get an embedding for a single string.
    ///
    /// # Arguments
    /// * `text` - The text chunk/string to generate an embedding for.
    ///
    /// # Errors
    /// * [`OpenAIError`] - If the request to OpenAI fails.
    ///  
    /// # Returns
    /// * `Result<(Chunk, Embedding), OpenAIError>` - A result containing
    /// a pair of the original text and the embedding that was generated.
    async fn generate_embedding(&self, text: Chunk) -> Result<(Chunk, Embedding), Self::ErrorType> {
        let request_body = EmbeddingRequest::builder()
            .input(text.clone().into())
            .model(self.embedding_model)
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
            OpenAIClient::handle_embedding_success_response(vec![text.clone()], embedding_response)
                [0]
            .clone(),
        )
    }
}

#[async_trait]
impl AsyncChatClient for OpenAIClient {
    type ErrorType = OpenAIError;
    /// # invoke
    /// Function to generate a chat completion for a vector of [`PromptMessage`].
    /// Allows you to get a chat completion for a vector of [`PromptMessage`].
    ///
    /// # Arguments
    /// * `prompt_messages` - The prompt messages to generate a chat completion for.
    ///
    /// # Errors
    /// * [`OpenAIError`] - If the request to OpenAI fails.
    ///  
    /// # Returns
    /// * `Result<PromptMessage, OpenAIError>` - A result containing
    /// the chat completion that was generated.
    async fn invoke(
        &self,
        prompt_messages: Vec<PromptMessage>,
    ) -> Result<PromptMessage, Self::ErrorType> {
        let chat_messages: Vec<ChatMessage> =
            prompt_messages.into_iter().map(ChatMessage::from).collect();

        let request_body: ChatCompletionRequest =  ChatCompletionRequest::builder()
            .model(OpenAIModel::Gpt4)
            .messages(chat_messages)
            .build();
        


        todo!()
    }
}

#[cfg(test)]
mod embedding_client_tests {
    use crate::clients::model::errors::{OpenAIError, OpenAIErrorBody, OpenAIErrorData};
    use crate::clients::openai_client::OpenAIClient;
    use crate::clients::traits::AsyncEmbeddingClient;
    use crate::common::embedding_shared::OpenAIEmbeddingModel;
    use crate::common::types::{Chunk, Chunks, Embedding};

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

    // Errors come in the same format just with different values,
    // so will use this for testing all error handling related
    //to errors with deserializable bodies
    const ERROR_RESPONSE: &'static str = r#"
    {
        "error": {
            "message": "Incorrect API key provided: fdas. You can find your API key at https://platform.openai.com/account/api-keys.",
            "type": "invalid_request_error",
            "param": null,
            "code": "invalid_api_key"
        }
    }
    "#;

    #[tokio::test]
    async fn test_correct_response_succeeds() {
        std::env::set_var("OPENAI_API_KEY", "fake key");
        let mut server = mockito::Server::new();
        let url = server.url();
        let model: OpenAIEmbeddingModel = OpenAIEmbeddingModel::TextEmbeddingAda002;
        let mut client: OpenAIClient = OpenAIClient::try_new(model).unwrap();
        client.url = url.clone();
        let mock = server
            .mock("POST", "/")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(EMBEDDING_RESPONSE)
            .create();
        let expected_embedding = Embedding::from(vec![
            -0.006929283495992422,
            -0.005336422007530928,
            -0.009327292,
            -0.024047505110502243,
        ]);
        // Test batch request
        let chunks: Chunks = Chunks::from(vec![Chunk::from("Test-0"), Chunk::from("Test-1")]);
        let response = client.generate_embeddings(chunks).await.unwrap();
        mock.assert();
        for (i, (chunk, embedding)) in response.into_iter().enumerate() {
            assert_eq!(chunk, Chunk::from(format!("Test-{}", i)));
            assert_eq!(embedding, expected_embedding);
        }
        // Test single request
        // This is not a great test as for a single request you would only get a vec of length 1
        // But the mocked response has two
        let chunk = Chunk::from("Test-0");
        let response = client.generate_embedding(chunk).await.unwrap();
        assert_eq!(response.0, Chunk::from("Test-0"));
        assert_eq!(response.1, expected_embedding);
    }

    #[tokio::test]
    async fn test_400_gives_correct_error() {
        std::env::set_var("OPENAI_API_KEY", "fake key");
        let mut server = mockito::Server::new();
        let url = server.url();
        let model: OpenAIEmbeddingModel = OpenAIEmbeddingModel::TextEmbeddingAda002;
        let mut client: OpenAIClient = OpenAIClient::try_new(model).unwrap();
        client.url = url.clone();
        let mock = server
            .mock("POST", "/")
            .with_status(400)
            .with_header("content-type", "application/json")
            .with_body(ERROR_RESPONSE)
            .create();
        let expected_response = OpenAIError::CODE400(OpenAIErrorBody {
            error: OpenAIErrorData {
                message: "Incorrect API key provided: fdas. You can find your API key at https://platform.openai.com/account/api-keys.".to_string(),
                error_type: "invalid_request_error".to_string(),
                param: None,
                code: "invalid_api_key".to_string()
            }
        });
        // Test batch request
        let chunks: Chunks = Chunks::from(vec![Chunk::from("Test-0"), Chunk::from("Test-1")]);
        let response = client.generate_embeddings(chunks).await.unwrap_err();
        mock.assert();
        assert_eq!(response, expected_response);
        // Test single request
        let chunk = Chunk::from("Test-0");
        let response = client.generate_embedding(chunk).await.unwrap_err();
        assert_eq!(response, expected_response);
    }

    #[tokio::test]
    async fn test_401_gives_correct_error() {
        std::env::set_var("OPENAI_API_KEY", "fake key");
        let mut server = mockito::Server::new();
        let url = server.url();
        let model: OpenAIEmbeddingModel = OpenAIEmbeddingModel::TextEmbeddingAda002;
        let mut client: OpenAIClient = OpenAIClient::try_new(model).unwrap();
        client.url = url.clone();
        let mock = server
            .mock("POST", "/")
            .with_status(401)
            .with_header("content-type", "application/json")
            .with_body(ERROR_RESPONSE)
            .create();
        let expected_response = OpenAIError::CODE401(OpenAIErrorBody {
            error: OpenAIErrorData {
                message: "Incorrect API key provided: fdas. You can find your API key at https://platform.openai.com/account/api-keys.".to_string(),
                error_type: "invalid_request_error".to_string(),
                param: None,
                code: "invalid_api_key".to_string()
            }
        });
        // Test batch request
        let chunks: Chunks = Chunks::from(vec![Chunk::from("Test-0"), Chunk::from("Test-1")]);
        let response = client.generate_embeddings(chunks).await.unwrap_err();
        mock.assert();
        assert_eq!(response, expected_response);
        // Test single request
        let chunk: Chunk = Chunk::from("Test-0");
        let response = client.generate_embedding(chunk).await.unwrap_err();
        assert_eq!(response, expected_response)
    }

    #[tokio::test]
    async fn test_429_gives_correct_error() {
        std::env::set_var("OPENAI_API_KEY", "fake key");
        let mut server = mockito::Server::new();
        let url = server.url();
        let model: OpenAIEmbeddingModel = OpenAIEmbeddingModel::TextEmbeddingAda002;
        let mut client: OpenAIClient = OpenAIClient::try_new(model).unwrap();
        client.url = url.clone();
        let mock = server
            .mock("POST", "/")
            .with_status(429)
            .with_header("content-type", "application/json")
            .with_body(ERROR_RESPONSE)
            .create();
        let expected_response = OpenAIError::CODE429(OpenAIErrorBody {
            error: OpenAIErrorData {
                message: "Incorrect API key provided: fdas. You can find your API key at https://platform.openai.com/account/api-keys.".to_string(),
                error_type: "invalid_request_error".to_string(),
                param: None,
                code: "invalid_api_key".to_string()
            }
        });
        // Test batch request
        let chunks: Chunks = Chunks::from(vec![Chunk::from("Test-0"), Chunk::from("Test-1")]);
        let response = client.generate_embeddings(chunks).await.unwrap_err();
        mock.assert();
        assert_eq!(response, expected_response);
        // Test single request
        let chunk: Chunk = Chunk::from("Test-0");
        let response = client.generate_embedding(chunk).await.unwrap_err();
        assert_eq!(response, expected_response);
    }

    #[tokio::test]
    async fn test_500_gives_correct_error() {
        std::env::set_var("OPENAI_API_KEY", "fake key");
        let mut server = mockito::Server::new();
        let url = server.url();
        let model: OpenAIEmbeddingModel = OpenAIEmbeddingModel::TextEmbeddingAda002;
        let mut client: OpenAIClient = OpenAIClient::try_new(model).unwrap();
        client.url = url.clone();
        let mock = server
            .mock("POST", "/")
            .with_status(500)
            .with_header("content-type", "application/json")
            .with_body(ERROR_RESPONSE)
            .create();
        let expected_response = OpenAIError::CODE500(OpenAIErrorBody {
            error: OpenAIErrorData {
                message: "Incorrect API key provided: fdas. You can find your API key at https://platform.openai.com/account/api-keys.".to_string(),
                error_type: "invalid_request_error".to_string(),
                param: None,
                code: "invalid_api_key".to_string()
            }
        });
        // Test batch request
        let chunks: Chunks = Chunks::from(vec![Chunk::from("Test-0"), Chunk::from("Test-1")]);
        let response = client.generate_embeddings(chunks).await.unwrap_err();
        mock.assert();
        assert_eq!(response, expected_response);
        // Test single request
        let chunk: Chunk = Chunk::from("Test-0");
        let response = client.generate_embedding(chunk).await.unwrap_err();
        assert_eq!(response, expected_response)
    }

    #[tokio::test]
    async fn test_503_gives_correct_error() {
        std::env::set_var("OPENAI_API_KEY", "fake key");
        let mut server = mockito::Server::new();
        let url = server.url();
        let model: OpenAIEmbeddingModel = OpenAIEmbeddingModel::TextEmbeddingAda002;
        let mut client: OpenAIClient = OpenAIClient::try_new(model).unwrap();
        client.url = url.clone();
        let mock = server
            .mock("POST", "/")
            .with_status(503)
            .with_header("content-type", "application/json")
            .with_body(ERROR_RESPONSE)
            .create();
        let expected_response = OpenAIError::CODE503(OpenAIErrorBody {
            error: OpenAIErrorData {
                message: "Incorrect API key provided: fdas. You can find your API key at https://platform.openai.com/account/api-keys.".to_string(),
                error_type: "invalid_request_error".to_string(),
                param: None,
                code: "invalid_api_key".to_string()
            }
        });
        // Test batch request
        let chunks: Chunks = Chunks::from(vec![Chunk::from("Test-0"), Chunk::from("Test-1")]);
        let response = client.generate_embeddings(chunks).await.unwrap_err();
        mock.assert();
        assert_eq!(response, expected_response);
        // Test single request
        let chunk: Chunk = Chunk::from("Test-0");
        let response = client.generate_embedding(chunk).await.unwrap_err();
        assert_eq!(response, expected_response)
    }

    #[tokio::test]
    async fn test_undefined_gives_correct_error() {
        std::env::set_var("OPENAI_API_KEY", "fake key");
        let mut server = mockito::Server::new();
        let url = server.url();
        let model: OpenAIEmbeddingModel = OpenAIEmbeddingModel::TextEmbeddingAda002;
        let mut client: OpenAIClient = OpenAIClient::try_new(model).unwrap();
        client.url = url.clone();
        let mock = server
            .mock("POST", "/")
            .with_status(409)
            .with_header("content-type", "application/json")
            .with_body(ERROR_RESPONSE)
            .create();
        let expected_response = OpenAIError::UNDEFINED(409, ERROR_RESPONSE.to_string());
        // Test batch request
        let chunks: Chunks = Chunks::from(vec![Chunk::from("Test-0"), Chunk::from("Test-1")]);
        let response = client.generate_embeddings(chunks).await.unwrap_err();
        mock.assert();
        assert_eq!(response, expected_response);
        // Test single request
        let chunk: Chunk = Chunk::from("Test-0");
        let response = client.generate_embedding(chunk).await.unwrap_err();
        assert_eq!(response, expected_response)
    }

    #[tokio::test]
    async fn test_bad_body_gives_correct_error() {
        std::env::set_var("OPENAI_API_KEY", "fake key");
        let mut server = mockito::Server::new();
        let url = server.url();
        let model: OpenAIEmbeddingModel = OpenAIEmbeddingModel::TextEmbeddingAda002;
        let mut client: OpenAIClient = OpenAIClient::try_new(model).unwrap();
        client.url = url.clone();
        let mock = server
            .mock("POST", "/")
            .with_status(401)
            .with_header("content-type", "application/json")
            .with_body("Sorry cant help right now")
            .create();
        let expected_response = OpenAIError::ErrorDeserializingResponseBody(
            401,
            "expected value at line 1 column 1".to_string(),
        );
        // Test batch request
        let chunks: Chunks = Chunks::from(vec![Chunk::from("Test-0"), Chunk::from("Test-1")]);
        let response = client.generate_embeddings(chunks).await.unwrap_err();
        mock.assert();
        assert_eq!(response, expected_response);
        // Test single request
        let chunk: Chunk = Chunk::from("Test-0");
        let response = client.generate_embedding(chunk).await.unwrap_err();
        assert_eq!(response, expected_response)
    }
}
