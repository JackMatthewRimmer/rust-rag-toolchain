use crate::clients::open_ai::model::embeddings::{
    BatchEmbeddingRequest, EmbeddingObject, EmbeddingRequest, EmbeddingResponse,
};
use crate::clients::open_ai::model::errors::OpenAIError;
use crate::clients::open_ai::open_ai_core::OpenAIHttpClient;
use crate::clients::traits::AsyncEmbeddingClient;
use crate::common::embedding_shared::OpenAIEmbeddingModel;
use crate::common::types::{Chunk, Chunks, Embedding};

use async_trait::async_trait;
use std::env::VarError;

const OPENAI_EMBEDDING_URL: &str = "https://api.openai.com/v1/embeddings";

/// # OpenAIEmbeddingClient
/// Allows for interacting with the OpenAI API to generate embeddings.
/// You can either embed a single string or a batch of strings.
///
/// # Required Environment Variables
/// OPENAI_API_KEY: The API key to use for the OpenAI API
pub struct OpenAIEmbeddingClient {
    url: String,
    client: OpenAIHttpClient,
    embedding_model: OpenAIEmbeddingModel,
}

impl OpenAIEmbeddingClient {
    pub fn try_new(
        embedding_model: OpenAIEmbeddingModel,
    ) -> Result<OpenAIEmbeddingClient, VarError> {
        let client: OpenAIHttpClient = OpenAIHttpClient::try_new()?;
        Ok(OpenAIEmbeddingClient {
            url: OPENAI_EMBEDDING_URL.into(),
            client,
            embedding_model,
        })
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
impl AsyncEmbeddingClient for OpenAIEmbeddingClient {
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

        let response: EmbeddingResponse = self.client.send_request(request_body, &self.url).await?;
        Ok(Self::handle_embedding_success_response(text, response))
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
        let response: EmbeddingResponse = self.client.send_request(request_body, &self.url).await?;
        Ok(Self::handle_embedding_success_response(vec![text], response)[0].clone())
    }
}

#[cfg(test)]
mod embedding_client_tests {
    use super::*;
    use crate::clients::open_ai::model::errors::{OpenAIErrorBody, OpenAIErrorData};

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
        let mut client: OpenAIEmbeddingClient = OpenAIEmbeddingClient::try_new(model).unwrap();
        client.url = url;
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
        let mut client: OpenAIEmbeddingClient = OpenAIEmbeddingClient::try_new(model).unwrap();
        client.url = url;
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
        let mut client: OpenAIEmbeddingClient = OpenAIEmbeddingClient::try_new(model).unwrap();
        client.url = url;
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
        let mut client: OpenAIEmbeddingClient = OpenAIEmbeddingClient::try_new(model).unwrap();
        client.url = url;
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
        let mut client: OpenAIEmbeddingClient = OpenAIEmbeddingClient::try_new(model).unwrap();
        client.url = url;
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
        let mut client: OpenAIEmbeddingClient = OpenAIEmbeddingClient::try_new(model).unwrap();
        client.url = url;
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
        let mut client: OpenAIEmbeddingClient = OpenAIEmbeddingClient::try_new(model).unwrap();
        client.url = url;
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
        let mut client: OpenAIEmbeddingClient = OpenAIEmbeddingClient::try_new(model).unwrap();
        client.url = url;
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
