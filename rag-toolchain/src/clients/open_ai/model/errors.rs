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

/// # [`OpenAIError`]
///
/// This error type largely mirrors the error codes list here
/// <https://platform.openai.com/docs/guides/error-codes>
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
    /// # Carries underlying error and the status code
    #[error("Error deserializining response body: status code = {0}, error = {1}")]
    ErrorDeserializingResponseBody(u16, String),
    /// # Carries underlying error if something went wrong when reading from a stream
    #[error("Error reading stream: {0}")]
    ErrorReadingStream(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_code_400_display() {
        let error_body = OpenAIErrorBody {
            error: OpenAIErrorData {
                message: "Invalid request".to_string(),
                error_type: "Bad Request".to_string(),
                param: None,
                code: "400".to_string(),
            },
        };
        let error = OpenAIError::CODE400(error_body.clone());
        let expected = format!("Bad Request: {:?}", error_body);
        assert_eq!(format!("{}", error), expected);
    }

    #[test]
    fn test_code_401_display() {
        let error_body = OpenAIErrorBody {
            error: OpenAIErrorData {
                message: "Invalid API key".to_string(),
                error_type: "Unauthorized".to_string(),
                param: None,
                code: "401".to_string(),
            },
        };
        let error = OpenAIError::CODE401(error_body.clone());
        let expected = format!(
            "Invalid Authentication or Incorrect API Key provided: {:?}",
            error_body
        );
        assert_eq!(format!("{}", error), expected);
    }

    #[test]
    fn test_code_429_display() {
        let error_body = OpenAIErrorBody {
            error: OpenAIErrorData {
                message: "Rate limit exceeded".to_string(),
                error_type: "Rate Limit Exceeded".to_string(),
                param: None,
                code: "429".to_string(),
            },
        };
        let error = OpenAIError::CODE429(error_body.clone());
        let expected = format!(
            "Rate limit reached or monthly quota exceeded: {:?}",
            error_body
        );
        assert_eq!(format!("{}", error), expected);
    }

    #[test]
    fn test_code_500_display() {
        let error_body = OpenAIErrorBody {
            error: OpenAIErrorData {
                message: "Internal server error".to_string(),
                error_type: "Internal Server Error".to_string(),
                param: None,
                code: "500".to_string(),
            },
        };
        let error = OpenAIError::CODE500(error_body.clone());
        let expected = format!("Server error: {:?}", error_body);
        assert_eq!(format!("{}", error), expected);
    }

    #[test]
    fn test_code_503_display() {
        let error_body = OpenAIErrorBody {
            error: OpenAIErrorData {
                message: "Engine overloaded".to_string(),
                error_type: "Service Unavailable".to_string(),
                param: None,
                code: "503".to_string(),
            },
        };
        let error = OpenAIError::CODE503(error_body.clone());
        let expected = format!("The engine is currently overloaded: {:?}", error_body);
        assert_eq!(format!("{}", error), expected);
    }

    #[test]
    fn test_undefined_display() {
        let status_code = 404;
        let error_message = "Not Found".to_string();
        let error = OpenAIError::Undefined(status_code, error_message.clone());
        let expected = format!("Undefined Error. This should not happen, if this is a missed error please report it: https://github.com/JackMatthewRimmer/rust-rag-toolchain: status code = {}, error = {}", status_code, error_message);
        assert_eq!(format!("{}", error), expected);
    }

    #[test]
    fn test_error_sending_request_display() {
        let error_message = "Connection timed out".to_string();
        let error = OpenAIError::ErrorSendingRequest(error_message.clone());
        let expected = format!("Error sending request: {}", error_message);
        assert_eq!(format!("{}", error), expected);
    }

    #[test]
    fn test_error_getting_response_body_display() {
        let error_message = "Failed to read response body".to_string();
        let error = OpenAIError::ErrorGettingResponseBody(error_message.clone());
        let expected = format!("Error getting response body: {}", error_message);
        assert_eq!(format!("{}", error), expected);
    }

    #[test]
    fn test_error_deserializing_response_body_display() {
        let status_code = 400;
        let error_message = "Invalid JSON format".to_string();
        let error = OpenAIError::ErrorDeserializingResponseBody(status_code, error_message.clone());
        let expected = format!(
            "Error deserializining response body: status code = {}, error = {}",
            status_code, error_message
        );
        assert_eq!(format!("{}", error), expected);
    }

    #[test]
    fn test_error_reading_stream_display() {
        let error_message = "Failed to read from stream".to_string();
        let error = OpenAIError::ErrorReadingStream(error_message.clone());
        let expected = format!("Error reading stream: {}", error_message);
        assert_eq!(format!("{}", error), expected);
    }
}
