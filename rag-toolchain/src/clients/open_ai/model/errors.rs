use serde::Deserialize;
use std::fmt::Display;

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

#[derive(Debug, PartialEq, Clone)]
pub enum OpenAIError {
    /// # Invalid Authentication or Incorrect API Key provided
    CODE400(OpenAIErrorBody),
    /// # Invalid Authentication or Incorrect API Key provided
    CODE401(OpenAIErrorBody),
    /// # Rate limit reached or Monthly quota exceeded
    CODE429(OpenAIErrorBody),
    /// # Server Error
    CODE500(OpenAIErrorBody),
    /// # The engine is currently overloaded
    CODE503(OpenAIErrorBody),
    /// # Missed cases for error codes, includes Status Code and Error Body as a string
    Undefined(u16, String),
    ErrorSendingRequest(String),
    /// # Carries underlying error
    ErrorGettingResponseBody(String),
    // # Carries underlying error and the status code
    ErrorDeserializingResponseBody(u16, String),
    ErrorReadingStream(String),
}

impl std::error::Error for OpenAIError {}
impl Display for OpenAIError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OpenAIError::CODE400(error_body) => {
                write!(f, "Bad Request: {}", error_body.error.message)
            }
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
            OpenAIError::Undefined(status_code, error_body) => {
                write!(f, "Undefined Error. This should not happen, if this is a missed error please report it: https://github.com/JackMatthewRimmer/rust-rag-toolchain: {} - {}", status_code, error_body)
            }
            OpenAIError::ErrorSendingRequest(error) => {
                write!(f, "Error Sending Request: {}", error)
            }
            OpenAIError::ErrorGettingResponseBody(error) => {
                write!(f, "Error Getting Response Body: {}", error)
            }
            OpenAIError::ErrorDeserializingResponseBody(code, error) => {
                write!(
                    f,
                    "Status Code: {} Error Deserializing Response Body: {}",
                    code, error
                )
            }
            OpenAIError::ErrorReadingStream(error) => {
                write!(f, "Error Reading Stream: {}", error)
            }
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn openai_error_code_429_display() {
        let error_body = with_open_ai_error_body();
        let error = OpenAIError::CODE429(error_body);
        assert_eq!(
            error.to_string(),
            "Rate limit reached or Monthly quota exceeded: error"
        );
    }

    #[test]
    fn openai_error_code_500_display() {
        let error_body = with_open_ai_error_body();
        let error = OpenAIError::CODE500(error_body);
        assert_eq!(error.to_string(), "Server Error: error");
    }

    #[test]
    fn openai_error_code_503_display() {
        let error_body = with_open_ai_error_body();
        let error = OpenAIError::CODE503(error_body);
        assert_eq!(
            error.to_string(),
            "The engine is currently overloaded: error"
        );
    }

    #[test]
    fn openai_error_undefined_display() {
        let error = OpenAIError::Undefined(404, "Not Found".to_string());
        assert_eq!(
            error.to_string(),
            "Undefined Error. This should not happen, if this is a missed error please report it: https://github.com/JackMatthewRimmer/rust-rag-toolchain: 404 - Not Found"
        );
    }

    #[test]
    fn openai_error_sending_request_display() {
        let error = OpenAIError::ErrorSendingRequest("Failed to send request".to_string());
        assert_eq!(
            error.to_string(),
            "Error Sending Request: Failed to send request"
        );
    }

    #[test]
    fn openai_error_getting_response_body_display() {
        let error =
            OpenAIError::ErrorGettingResponseBody("Failed to get response body".to_string());
        assert_eq!(
            error.to_string(),
            "Error Getting Response Body: Failed to get response body"
        );
    }

    #[test]
    fn openai_error_deserializing_response_body_display() {
        let error = OpenAIError::ErrorDeserializingResponseBody(
            500,
            "Failed to deserialize response body".to_string(),
        );
        assert_eq!(
            error.to_string(),
            "Status Code: 500 Error Deserializing Response Body: Failed to deserialize response body"
        );
    }

    #[test]
    fn openai_error_reading_stream_display() {
        let error = OpenAIError::ErrorReadingStream("Failed to read stream".to_string());
        assert_eq!(
            error.to_string(),
            "Error Reading Stream: Failed to read stream"
        );
    }

    fn with_open_ai_error_body() -> OpenAIErrorBody {
        let data = OpenAIErrorData {
            param: None,
            message: "error".into(),
            code: "error".into(),
            error_type: "error".into(),
        };

        OpenAIErrorBody { error: data }
    }
}
