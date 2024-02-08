use crate::clients::open_ai::model::errors::{OpenAIError, OpenAIErrorBody};

use dotenv::dotenv;
use reqwest::header::{HeaderValue, CONTENT_TYPE};
use reqwest::{Client, RequestBuilder, Response, StatusCode};
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::env;
use std::env::VarError;

pub struct OpenAIHttpClient {
    client: Client,
    api_key: String,
}

impl OpenAIHttpClient {
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
    pub fn try_new() -> Result<OpenAIHttpClient, VarError> {
        dotenv().ok();
        let api_key: String = match env::var::<String>("OPENAI_API_KEY".into()) {
            Ok(api_key) => api_key,
            Err(e) => return Err(e),
        };
        let client: Client = Client::new();
        Ok(OpenAIHttpClient { api_key, client })
    }

    // # send_reqest
    // Sends a request to the OpenAI API and returns the response
    //
    // # Arguments
    // * `body` - The body of the request
    // * `url` - The url to send the request to
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

    pub async fn send_request<T, U>(&self, body: T, url: &str) -> Result<U, OpenAIError>
    where
        T: Serialize,
        U: DeserializeOwned,
    {
        let request = self.build_requeset(body, url);
        let response: reqwest::Response = request
            .send()
            .await
            .map_err(|error| OpenAIError::ErrorSendingRequest(error.to_string()))?;

        let status_code: StatusCode = response.status();

        if !status_code.is_success() {
            let mapped_error: OpenAIError = Self::handle_error_response(response).await;
            return Err(mapped_error);
        }

        let response_body: String = response
            .text()
            .await
            .map_err(|error| OpenAIError::ErrorGettingResponseBody(error.to_string()))?;

        serde_json::from_str(&response_body).map_err(|error| {
            OpenAIError::ErrorDeserializingResponseBody(status_code.as_u16(), error.to_string())
        })
    }

    /// # build_request
    ///
    /// Helper method to build a request with the correct headers and body
    fn build_requeset<T>(&self, request_body: T, url: &str) -> RequestBuilder
    where
        T: Serialize,
    {
        let content_type = HeaderValue::from_static("application/json");
        self.client
            .post(url)
            .bearer_auth(self.api_key.clone())
            .header(CONTENT_TYPE, content_type)
            .json(&request_body)
    }

    // # handle_error_response

    // Explicit error mapping between response codes and error types
    //
    // # Arguments
    // `response` - The reqwest response from OpenAI
    //
    // # Returns
    // `OpenAIError` - The error type that maps to the response code
    async fn handle_error_response(response: Response) -> OpenAIError {
        // Map response objects into some form of enum error
        let status_code = response.status().as_u16();
        let body_text = match response.text().await {
            Ok(text) => text,
            Err(e) => return OpenAIError::Undefined(status_code, e.to_string()),
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
            undefined => OpenAIError::Undefined(undefined, body_text),
        }
    }
}
