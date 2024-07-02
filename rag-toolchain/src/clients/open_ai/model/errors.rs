use serde::Deserialize;
use thiserror::Error;

// This is what is returned from OpenAI
// when an error occurs
#[derive(Debug, Deserialize, PartialEq, Clone)]
pub struct OpenAIErrorBody {
    pub error: OpenAIErrorData,
}

#[derive(Debug, Deserialize, PartialEq, Clone)]
pub struct OpenAIErrorData {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    pub param: Option<String>,
    pub code: String,
}
// --------------------------------------------------------------------------------

#[derive(Error, Debug, PartialEq, Clone)]
pub enum OpenAIError {
    /// # Invalid Authentication or Incorrect API Key provided
    #[error("Bad Request: {0:?}")]
    CODE400(OpenAIErrorBody),
    /// # Invalid Authentication or Incorrect API Key provided
    #[error("Invalid Authentication or Incorrect API Key provided: {0:?}")]
    CODE401(OpenAIErrorBody),
    /// # Rate limit reached or Monthly quota exceeded
    #[error("Rate limit reached or monthly quota exceeded: {0:?}")]
    CODE429(OpenAIErrorBody),
    /// # Server Error
    #[error("Server error: {0:?}")]
    CODE500(OpenAIErrorBody),
    /// # The engine is currently overloaded
    #[error("The engine is currently overloaded: {0:?}")]
    CODE503(OpenAIErrorBody),
    /// # Missed cases for error codes, includes Status Code and Error Body as a string
    #[error("Undefined Error. This should not happen, if this is a missed error please report it: https://github.com/JackMatthewRimmer/rust-rag-toolchain: status code = {0}, error = {1}")]
    Undefined(u16, String),
    #[error("Error sending request: {0}")]
    ErrorSendingRequest(String),
    /// # Carries underlying error
    #[error("Error getting response body: {0}")]
    ErrorGettingResponseBody(String),
    // # Carries underlying error and the status code
    #[error("Error deserializining response body: status code = {0}, error = {1} ")]
    ErrorDeserializingResponseBody(u16, String),
    #[error("Error reading stream: {0}")]
    ErrorReadingStream(String),
}
