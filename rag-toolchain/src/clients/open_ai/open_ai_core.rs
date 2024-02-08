use crate::clients::open_ai::model::errors::{OpenAIError, OpenAIErrorBody};

use dotenv::dotenv;
use reqwest::header::{HeaderValue, CONTENT_TYPE};
use reqwest::{Client, RequestBuilder, Response, StatusCode};
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::env;
use std::env::VarError;

#[derive(Debug)]
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

#[cfg(test)]
mod open_ai_core_tests {
    use super::*;
    use mockito::{Mock, Server, ServerGuard};
    use serde::{de::IntoDeserializer, Deserialize, Serialize};

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

    #[test]
    fn status_400_maps_correctly() {
        let expected_error_body = serde_json::from_str(ERROR_RESPONSE).unwrap();
        let expected_error = OpenAIError::CODE400(expected_error_body);
        assert_status_mapping(400, expected_error);
    }

    #[test]
    fn status_401_maps_correctly() {
        let expected_error_body = serde_json::from_str(ERROR_RESPONSE).unwrap();
        let expected_error = OpenAIError::CODE401(expected_error_body);
        assert_status_mapping(401, expected_error);
    }

    #[test]
    fn status_429_maps_correctly() {
        let expected_error_body = serde_json::from_str(ERROR_RESPONSE).unwrap();
        let expected_error = OpenAIError::CODE429(expected_error_body);
        assert_status_mapping(429, expected_error);
    }

    #[test]
    fn status_500_maps_correctly() {
        let expected_error_body = serde_json::from_str(ERROR_RESPONSE).unwrap();
        let expected_error = OpenAIError::CODE500(expected_error_body);
        assert_status_mapping(500, expected_error);
    }

    #[test]
    fn status_503_maps_correctly() {
        let expected_error_body = serde_json::from_str(ERROR_RESPONSE).unwrap();
        let expected_error = OpenAIError::CODE503(expected_error_body);
        assert_status_mapping(503, expected_error);
    }

    #[test]
    fn undefined_maps_correctly() {
        let expected_error = OpenAIError::Undefined(404, ERROR_RESPONSE.into());
        assert_status_mapping(404, expected_error);
    }

    #[tokio::test]
    async fn error_deserializing_response_body_maps_correctly() {
        let response_body = "some invalid response";
        let status_code: u16 = 400;
        let body = RequestBody {
            message: "hello".into(),
        };
        let (client, mut server) = with_mocked_client();
        let mock = with_mocked_request(&mut server, status_code.into(), response_body);
        let error = client
            .send_request::<RequestBody, RequestBody>(body, &server.url())
            .await
            .unwrap_err();
        // The error here is the message from serde
        let expected_error = OpenAIError::ErrorDeserializingResponseBody(
            status_code,
            "expected value at line 1 column 1".into(),
        );
        mock.assert();
        assert_eq!(expected_error, error);
    }

    // Helper method to assert all known status codes are mapped correctly
    fn assert_status_mapping(status_code: usize, expected_error: OpenAIError) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let body = RequestBody {
            message: "hello".into(),
        };
        let (client, mut server) = with_mocked_client();
        let mock = with_mocked_request(&mut server, status_code, ERROR_RESPONSE);
        let error = rt
            .block_on(client.send_request::<RequestBody, RequestBody>(body, &server.url()))
            .unwrap_err();
        mock.assert();
        assert_eq!(expected_error, error);
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
    fn with_mocked_client() -> (OpenAIHttpClient, ServerGuard) {
        std::env::set_var("OPENAI_API_KEY", "fake key");
        let server = Server::new();
        let client = OpenAIHttpClient::try_new().unwrap();
        (client, server)
    }

    #[derive(Debug, Serialize, Deserialize)]
    struct RequestBody {
        message: String,
    }
}
