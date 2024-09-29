use crate::clients::anthropic::model::errors::{AnthropicError, AnthropicErrorBody};

use dotenv::dotenv;
use reqwest::header::{HeaderValue, CONTENT_TYPE};
use reqwest::{Client, RequestBuilder, Response, StatusCode};
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::env;
use std::env::VarError;

const API_KEY_HEADER: &str = "x-api-key";
const API_VERSION_HEADER: &str = "anthropic-version";
const API_VERSION: &str = "2023-06-01";

#[derive(Debug)]
pub struct AnthropicHttpClient {
    client: Client,
    api_key: String,
}

impl AnthropicHttpClient {
    /// # [`AnthropicHttpClient::try_new`]
    /// Must have the ANTHROPIC_API_KEY environment variable set
    ///
    /// # Errors
    /// * [`VarError`] - If the ANTHROPIC_API_KEY environment variable is not set
    ///
    /// # Returns
    /// * [`AnthropicHttpClient`] - The newly created AnthropicHttpClient
    pub fn try_new() -> Result<AnthropicHttpClient, VarError> {
        dotenv().ok();
        let api_key: String = match env::var::<String>("ANTHROPIC_API_KEY".into()) {
            Ok(api_key) => api_key,
            Err(e) => return Err(e),
        };
        let client: Client = Client::new();
        Ok(AnthropicHttpClient { api_key, client })
    }

    /// # [`AnthropicHttpClient::send_request`]
    /// Sends a request to the Anthropic API and returns the response
    ///
    /// # Arguments
    /// * `body` - The body of the request
    /// * `url` - The url to send the request to
    ///
    /// # Errors
    /// * [`AnthropicError::ErrorSendingRequest`] - if request.send() errors
    /// * [`AnthropicError::ErrorGettingResponseBody`] - if response.text() errors
    /// * [`AnthropicError::ErrorDeserializingResponseBody`] - if serde_json::from_str() errors
    /// * [`AnthropicError`] - if the response code is not 200 this can be any of the associates status
    ///    code errors or variatn of `AnthropicError::UNDEFINED`
    ///
    /// # Returns
    /// [`U`] - The deserialized response from Anthropic
    pub async fn send_request<T, U>(&self, body: T, url: &str) -> Result<U, AnthropicError>
    where
        T: Serialize,
        U: DeserializeOwned,
    {
        let request = self.build_requeset(body, url);
        let response: reqwest::Response = request
            .send()
            .await
            .map_err(|error| AnthropicError::ErrorSendingRequest(error.to_string()))?;

        let status_code: StatusCode = response.status();

        if !status_code.is_success() {
            let mapped_error: AnthropicError = Self::handle_error_response(response).await;
            return Err(mapped_error);
        }

        let response_body: String = response
            .text()
            .await
            .map_err(|error| AnthropicError::ErrorGettingResponseBody(error.to_string()))?;

        serde_json::from_str(&response_body).map_err(|error| {
            AnthropicError::ErrorDeserializingResponseBody(status_code.as_u16(), error.to_string())
        })
    }

    /// # [`AnthropicHttpClient::build_requeset`]
    ///
    /// Helper method to build a request with the correct headers and body
    /// We are required to set a speceific API version on each request
    fn build_requeset<T>(&self, request_body: T, url: &str) -> RequestBuilder
    where
        T: Serialize,
    {
        let content_type = HeaderValue::from_static("application/json");
        self.client
            .post(url)
            .header(API_KEY_HEADER, self.api_key.clone())
            .header(API_VERSION_HEADER, API_VERSION)
            .header(CONTENT_TYPE, content_type)
            .json(&request_body)
    }

    /// # [`AnthropicHttpClient::handle_error_response`]
    ///
    /// Explicit error mapping between response codes and error types
    ///
    /// # Arguments
    /// `response` - The reqwest response from Anthropic
    ///
    /// # Returns
    /// [`AnthropicError`] - The error type that maps to the response code
    async fn handle_error_response(response: Response) -> AnthropicError {
        // Map response objects into some form of enum error
        let status_code = response.status().as_u16();
        let body_text = match response.text().await {
            Ok(text) => text,
            Err(e) => return AnthropicError::Undefined(status_code, e.to_string()),
        };

        let error_body: AnthropicErrorBody = match serde_json::from_str(&body_text) {
            Ok(error_body) => error_body,
            Err(e) => {
                return AnthropicError::ErrorDeserializingResponseBody(status_code, e.to_string())
            }
        };
        match status_code {
            400 => AnthropicError::CODE400(error_body),
            401 => AnthropicError::CODE401(error_body),
            403 => AnthropicError::CODE403(error_body),
            404 => AnthropicError::CODE404(error_body),
            413 => AnthropicError::CODE413(error_body),
            429 => AnthropicError::CODE429(error_body),
            500 => AnthropicError::CODE500(error_body),
            503 => AnthropicError::CODE503(error_body),
            undefined => AnthropicError::Undefined(undefined, body_text),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mockito::{Mock, Server, ServerGuard};
    use serde::{Deserialize, Serialize};

    const ERROR_RESPONSE: &'static str = r#"
    {
        "type": "error",
        "error": {
            "type": "not_found_error",
            "message": "The requested resource could not be found."
        }
    } 
    "#;

    #[test]
    fn missing_env_var_returns_error() {
        std::env::remove_var("ANTHROPIC_API_KEY");
        let error = AnthropicHttpClient::try_new().unwrap_err();
        assert_eq!(VarError::NotPresent, error);
    }

    #[tokio::test]
    async fn status_400_maps_correctly() {
        let expected_error_body = serde_json::from_str(ERROR_RESPONSE).unwrap();
        let expected_error = AnthropicError::CODE400(expected_error_body);
        assert_status_mapping(400, expected_error).await;
    }

    #[tokio::test]
    async fn status_401_maps_correctly() {
        let expected_error_body = serde_json::from_str(ERROR_RESPONSE).unwrap();
        let expected_error = AnthropicError::CODE401(expected_error_body);
        assert_status_mapping(401, expected_error).await;
    }

    #[tokio::test]
    async fn status_403_maps_correctly() {
        let expected_error_body = serde_json::from_str(ERROR_RESPONSE).unwrap();
        let expected_error = AnthropicError::CODE403(expected_error_body);
        assert_status_mapping(403, expected_error).await;
    }

    #[tokio::test]
    async fn status_404_maps_correctly() {
        let expected_error_body = serde_json::from_str(ERROR_RESPONSE).unwrap();
        let expected_error = AnthropicError::CODE404(expected_error_body);
        assert_status_mapping(404, expected_error).await;
    }

    #[tokio::test]
    async fn status_413_maps_correctly() {
        let expected_error_body = serde_json::from_str(ERROR_RESPONSE).unwrap();
        let expected_error = AnthropicError::CODE413(expected_error_body);
        assert_status_mapping(413, expected_error).await;
    }

    #[tokio::test]
    async fn status_429_maps_correctly() {
        let expected_error_body = serde_json::from_str(ERROR_RESPONSE).unwrap();
        let expected_error = AnthropicError::CODE429(expected_error_body);
        assert_status_mapping(429, expected_error).await;
    }

    #[tokio::test]
    async fn status_500_maps_correctly() {
        let expected_error_body = serde_json::from_str(ERROR_RESPONSE).unwrap();
        let expected_error = AnthropicError::CODE500(expected_error_body);
        assert_status_mapping(500, expected_error).await;
    }

    #[tokio::test]
    async fn status_503_maps_correctly() {
        let expected_error_body = serde_json::from_str(ERROR_RESPONSE).unwrap();
        let expected_error = AnthropicError::CODE503(expected_error_body);
        assert_status_mapping(503, expected_error).await;
    }

    #[tokio::test]
    async fn undefined_maps_correctly() {
        let expected_error = AnthropicError::Undefined(469, ERROR_RESPONSE.into());
        assert_status_mapping(469, expected_error).await;
    }

    #[tokio::test]
    async fn error_deserializing_response_body_maps_correctly() {
        let response_body = "some invalid response";
        let status_code: u16 = 400;
        let body = RequestBody {
            message: "hello".into(),
        };
        let (client, mut server) = with_mocked_client().await;
        let mock = with_mocked_request(&mut server, status_code.into(), response_body);
        let error = client
            .send_request::<RequestBody, RequestBody>(body, &server.url())
            .await
            .unwrap_err();
        // The error here is the message from serde
        let expected_error = AnthropicError::ErrorDeserializingResponseBody(
            status_code,
            "expected value at line 1 column 1".into(),
        );
        mock.assert();
        assert_eq!(expected_error, error);
    }

    // Helper method to assert all known status codes are mapped correctly
    async fn assert_status_mapping(status_code: usize, expected_error: AnthropicError) {
        let body = RequestBody {
            message: "hello".into(),
        };
        let (client, mut server) = with_mocked_client().await;
        let mock = with_mocked_request(&mut server, status_code, ERROR_RESPONSE);
        let error: AnthropicError = client
            .send_request::<RequestBody, RequestBody>(body, &server.url())
            .await
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
    async fn with_mocked_client() -> (AnthropicHttpClient, ServerGuard) {
        std::env::set_var("ANTHROPIC_API_KEY", "fake key");
        let server = Server::new_async().await;
        let client = AnthropicHttpClient::try_new().unwrap();
        (client, server)
    }

    #[derive(Debug, Serialize, Deserialize)]
    struct RequestBody {
        message: String,
    }
}
