use serde::Deserialize;
use thiserror::Error;

#[derive(Debug, Deserialize, PartialEq, Clone)]
pub struct AnthropicErrorBody {
    pub r#type: String,
    pub error: AnthropicErrorDetails,
}

#[derive(Debug, Deserialize, PartialEq, Clone)]
pub struct AnthropicErrorDetails {
    pub r#type: String,
    pub message: String,
}

/// # [` AnthropicError`]
///
/// This error type largely mirrors the error codes list here
/// <https://docs.anthropic.com/en/api/errors>.
#[derive(Error, Debug, PartialEq, Clone)]
pub enum AnthropicError {
    /// # There was an issue with the format or content of your request.
    /// # We may also use this error type for other 4XX status codes not listed below.
    #[error("Invalid Request Error: {0:?}")]
    CODE400(AnthropicErrorBody),
    /// # There’s an issue with your API key
    #[error("Authentication Error: {0:?}")]
    CODE401(AnthropicErrorBody),
    /// # Your API key does not have permission to use the specified resource.
    #[error("Permission Error: {0:?}")]
    CODE403(AnthropicErrorBody),
    /// # The requested resource was not found.
    #[error("Not Found Error: {0:?}")]
    CODE404(AnthropicErrorBody),
    /// # Request exceeds the maximum allowed number of bytes.
    #[error("Request Too Large Error: {0:?}")]
    CODE413(AnthropicErrorBody),
    /// # Your account has hit a rate limit.
    #[error("Rate Limit Error: {0:?}")]
    CODE429(AnthropicErrorBody),
    /// # An unexpected error has occurred internal to Anthropic’s systems
    #[error("API Error: {0:?}")]
    CODE500(AnthropicErrorBody),
    /// # Anthropic’s API is temporarily overloaded.
    #[error("Overloaded Error: {0:?}")]
    CODE503(AnthropicErrorBody),
    /// # Missed cases for error codes, includes Status Code and Error Body as a string. These can also represent internal logic errors.
    #[error("Undefined Error. This should not happen, if this is a missed error please report it: https://github.com/JackMatthewRimmer/rust-rag-toolchain: status code = {0}, error = {1}")]
    Undefined(u16, String),
    /// # Carries underlying error that may have occurred during sending the request
    #[error("Error sending request: {0}")]
    ErrorSendingRequest(String),
    /// # Carries underlying error that may have occured when trying to get the response body
    #[error("Error getting response body: {0}")]
    ErrorGettingResponseBody(String),
    // # Carries underlying error and the status code
    #[error("Error deserializining response body: status code = {0}, error = {1}")]
    ErrorDeserializingResponseBody(u16, String),
}
