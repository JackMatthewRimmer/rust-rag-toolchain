use crate::common::embedding_shared::OpenAIEmbeddingModel;
use crate::clients::open_ai::open_ai_core::OpenAIHttpClient;
use crate::clients::model::errors::OpenAIError;
use crate::clients::traits::AsyncEmbeddingClient;
use crate::common::types::{Chunk, Chunks, Embedding};
use crate::clients::model::embeddings::{
    BatchEmbeddingRequest, EmbeddingRequest, EmbeddingResponse, EmbeddingObject
};

use std::env::VarError;
use async_trait::async_trait;



const OPENAI_EMBEDDING_URL: &str = "https://api.openai.com/v1/embeddings";

/// # OpenAIEmbeddingClient
/// Allows for interacting with the OpenAI API to generate embeddings.
/// You can either embed a single string or a batch of strings.
///
/// # Required Environment Variables
/// OPENAI_API_KEY: The API key to use for the OpenAI API
struct OpenAIEmbeddingClient {
    url: &'static str,
    client: OpenAIHttpClient,
    embedding_model: OpenAIEmbeddingModel,
}

impl OpenAIEmbeddingClient {
    fn try_new(embedding_model: OpenAIEmbeddingModel) -> Result<OpenAIEmbeddingClient, VarError> {
        let client: OpenAIHttpClient = OpenAIHttpClient::try_new()?;
        Ok(OpenAIEmbeddingClient {
            url: OPENAI_EMBEDDING_URL,
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

        let response: EmbeddingResponse = self.client
            .send_request(request_body, self.url.clone())
            .await?;
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
        let response: EmbeddingResponse = self.client
            .send_request(request_body, self.url)
            .await?;
        Ok(Self::handle_embedding_success_response(vec![text], response)[0].clone())
    }
}