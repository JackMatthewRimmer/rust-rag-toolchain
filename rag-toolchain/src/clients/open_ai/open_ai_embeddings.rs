use crate::clients::open_ai::model::embeddings::{
    BatchEmbeddingRequest, EmbeddingObject, EmbeddingRequest, EmbeddingResponse,
};
use crate::clients::open_ai::model::errors::OpenAIError;
use crate::clients::open_ai::open_ai_core::OpenAIHttpClient;
use crate::clients::traits::AsyncEmbeddingClient;
use crate::common::{Chunk, Chunks, Embedding, OpenAIEmbeddingModel};
use std::env::VarError;

const OPENAI_EMBEDDING_URL: &str = "https://api.openai.com/v1/embeddings";

/// # [`OpenAIEmbeddingClient`]
/// Allows for interacting with the OpenAI API to generate embeddings.
/// You can either embed a single string or a batch of strings.
///
/// # Examples
/// ```
/// use rag_toolchain::common::*;
/// use rag_toolchain::clients::*;
/// async fn generate_embedding() {
///     let client: OpenAIEmbeddingClient = OpenAIEmbeddingClient::try_new(OpenAIEmbeddingModel::TextEmbeddingAda002).unwrap();
///     let chunk: Chunk = Chunk::new("this would be the text you are embedding");
///     let embedding: Embedding = client.generate_embedding(chunk).await.unwrap();
///     // This would be the vector representation of the text
///     let vector: Vec<f32> = embedding.vector();
/// }
/// ```
/// # Required Environment Variables
/// OPENAI_API_KEY: The API key to use for the OpenAI API
pub struct OpenAIEmbeddingClient {
    url: String,
    client: OpenAIHttpClient,
    embedding_model: OpenAIEmbeddingModel,
}

impl OpenAIEmbeddingClient {
    /// # [`OpenAIEmbeddingClient::try_new`]
    /// Constructor to create a new OpenAIEmbeddingClient.
    /// This will fail if the OPENAI_API_KEY environment variable is not set.
    ///
    /// # Arguments
    /// * `embedding_model`: [`OpenAIEmbeddingModel`] - The model to use for the embeddings
    ///
    /// # Errors
    /// * [`VarError`] - If the OPENAI_API_KEY environment variable is not set.
    ///
    /// # Returns
    /// * [`OpenAIEmbeddingClient`] - The newly created OpenAIEmbeddingClient
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

    /// # [`OpenAIEmbeddingClient::handle_embedding_success_response`]
    /// Takes a successful response and maps it into a vector of string embedding pairs
    /// assumption made the two iters will zip up 1:1 (as this should be the case)
    ///
    /// # Arguments
    /// *`input_text`: [`Chunks`] - The input text that was sent to OpenAI
    /// *`response`: [`EmbeddingResponse`] - The deserialized response from OpenAI
    ///
    /// # Returns
    /// [`Vec<Embedding>`] - A vector of string embedding pairs the can be stored
    fn handle_embedding_success_response(
        input_text: Chunks,
        response: EmbeddingResponse,
    ) -> Vec<Embedding> {
        // Map response objects into string embedding pairs
        let embedding_objects: Vec<EmbeddingObject> = response.data;
        embedding_objects
            .into_iter()
            .zip(input_text)
            .map(|(embedding_object, chunk)| Embedding::new(chunk, embedding_object.embedding))
            .collect()
    }
}

impl AsyncEmbeddingClient for OpenAIEmbeddingClient {
    type ErrorType = OpenAIError;

    /// # [`OpenAIEmbeddingClient::generate_embeddings`]
    /// Function to generate embeddings for [`Chunks`].
    /// Allows you to get an embedding for multiple strings.
    ///
    /// # Arguments
    /// * `text`: [`Chunk`] - The text chunks/strings to generate an embeddings for.
    ///
    /// # Errors
    /// * [`OpenAIError`] - If the request to OpenAI fails.
    ///
    /// # Returns
    /// * [`Vec<Embedding>`] - A result containing
    ///     pairs of the original text and the embedding that was generated.
    async fn generate_embeddings(&self, text: Chunks) -> Result<Vec<Embedding>, OpenAIError> {
        let input_text: Vec<String> = text
            .iter()
            .map(|chunk| (*chunk).content().to_string())
            .collect();

        let request_body = BatchEmbeddingRequest::builder()
            .input(input_text)
            .model(self.embedding_model)
            .build();

        let response: EmbeddingResponse = self.client.send_request(request_body, &self.url).await?;
        Ok(Self::handle_embedding_success_response(text, response))
    }

    /// # [`OpenAIEmbeddingClient::generate_embedding`]
    /// Function to generate an embedding for a [`Chunk`].
    /// Allows you to get an embedding for a single string.
    ///
    /// # Arguments
    /// * `text`: [`Chunk`] - The text chunk/string to generate an embedding for.
    ///
    /// # Errors
    /// * [`OpenAIError`] - If the request to OpenAI fails.
    ///
    /// # Returns
    /// * [`Embedding`] - the generated embedding
    async fn generate_embedding(&self, text: Chunk) -> Result<Embedding, Self::ErrorType> {
        let request_body = EmbeddingRequest::builder()
            .input(text.content().to_string())
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
    use mockito::{Mock, Server, ServerGuard};

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
        let (client, mut server) = with_mocked_client().await;
        let mock = with_mocked_request(&mut server, 200, EMBEDDING_RESPONSE);
        let expected_embedding = vec![
            -0.006929283495992422,
            -0.005336422007530928,
            -0.009327292,
            -0.024047505110502243,
        ];
        // Test batch request
        let chunks: Chunks = vec![Chunk::new("Test-0"), Chunk::new("Test-1")];
        let response = client.generate_embeddings(chunks).await.unwrap();
        mock.assert();
        for (i, embedding) in response.into_iter().enumerate() {
            assert_eq!(*embedding.chunk(), Chunk::new(format!("Test-{}", i)));
            assert_eq!(embedding.vector(), expected_embedding);
        }
        // Test single request
        // This is not a great test as for a single request you would only get a vec of length 1
        // But the mocked response has two
        let chunk = Chunk::new("Test-0");
        let response = client.generate_embedding(chunk).await.unwrap();
        assert_eq!(*response.chunk(), Chunk::new("Test-0"));
        assert_eq!(response.vector(), expected_embedding);
    }

    #[tokio::test]
    async fn test_400_gives_correct_error() {
        let (client, mut server) = with_mocked_client().await;
        let mock = with_mocked_request(&mut server, 400, ERROR_RESPONSE);
        let expected_response = OpenAIError::CODE400(OpenAIErrorBody {
            error: OpenAIErrorData {
                message: "Incorrect API key provided: fdas. You can find your API key at https://platform.openai.com/account/api-keys.".to_string(),
                error_type: "invalid_request_error".to_string(),
                param: None,
                code: "invalid_api_key".to_string()
            }
        });
        // Test batch request
        let chunks: Chunks = vec![Chunk::new("Test-0"), Chunk::new("Test-1")];
        let response = client.generate_embeddings(chunks).await.unwrap_err();
        mock.assert();
        assert_eq!(response, expected_response);
        // Test single request
        let chunk = Chunk::new("Test-0");
        let response = client.generate_embedding(chunk).await.unwrap_err();
        assert_eq!(response, expected_response);
    }

    // Method which mocks the response the server will give. this
    // allows us to stub the requests instead of sending them to OpenAI
    fn with_mocked_request(
        server: &mut ServerGuard,
        status_code: usize,
        response_body: &str,
    ) -> Mock {
        server
            .mock("POST", "/")
            .with_status(status_code)
            .with_header("content-type", "application/json")
            .with_body(response_body)
            .create()
    }

    // This methods returns a client which is pointing at the mocked url
    // and the mock server which we can orchestrate the stubbings on.
    async fn with_mocked_client() -> (OpenAIEmbeddingClient, ServerGuard) {
        std::env::set_var("OPENAI_API_KEY", "fake key");
        let server = Server::new_async().await;
        let url = server.url();
        let model = OpenAIEmbeddingModel::TextEmbeddingAda002;
        let mut client = OpenAIEmbeddingClient::try_new(model).unwrap();
        client.url = url;
        (client, server)
    }
}
