use serde::{Deserialize, Serialize};
use typed_builder::TypedBuilder;

#[derive(Debug, Serialize, Deserialize, PartialEq, TypedBuilder)]
#[serde(rename_all = "snake_case")]
pub struct MessagesRequest {
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub system: String,
    pub model: AnthropicModel,
    pub max_tokens: usize,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub stop_sequences: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<usize>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, Clone)]
pub struct MessagesResponse {
    pub id: String,
    pub r#type: String,
    pub role: Role,
    pub content: Vec<Content>,
    pub model: String,
    pub stop_reason: Option<StopReason>,
    pub stop_sequence: Option<String>,
    pub usage: Usage,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, Clone)]
pub struct Usage {
    pub input_tokens: usize,
    pub output_tokens: usize,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, Clone)]
#[serde(rename_all = "snake_case")]
pub enum AnthropicModel {
    #[serde(rename = "claude-3-5-sonnet-20240620")]
    Claude3Point5Sonnet,
    #[serde(rename = "claude-3-opus-20240229")]
    Claude3Opus,
    #[serde(rename = "claude-3-sonnet-20240229")]
    Claude3Sonnet,
    #[serde(rename = "claude-3-haiku-20240307")]
    Claude3Haiku,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, Clone)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    EndTurn,
    MaxTokens,
    StopSequence,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, Clone)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum Content {
    // Note only support of text here. Images need to be added later
    Text { text: String },
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, Clone)]
#[serde(rename_all = "snake_case")]
pub struct Message {
    pub role: Role,
    pub content: Vec<Content>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, Clone)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    User,
    Assistant,
}

#[cfg(test)]
mod request_model_tests {
    use super::*;

    const CHAT_MESSAGE_RESPONSE: &str = r#"
    {
        "id": "msg_01XFDUDYJgAACzvnptvVoYEL",
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "Hello!"
            }
        ],
        "model": "claude-3-5-sonnet-20240620",
        "stop_reason": "end_turn",
        "stop_sequence": null,
        "usage": {
            "input_tokens": 12,
            "output_tokens": 6
        }
    }
    "#;

    #[test]
    fn test_deserialize_chat_message_response() {
        let response: MessagesResponse = serde_json::from_str(CHAT_MESSAGE_RESPONSE).unwrap();

        let expected_response = MessagesResponse {
            id: "msg_01XFDUDYJgAACzvnptvVoYEL".to_string(),
            r#type: "message".to_string(),
            role: Role::Assistant,
            content: vec![Content::Text {
                text: "Hello!".to_string(),
            }],
            model: "claude-3-5-sonnet-20240620".to_string(),
            stop_reason: Some(StopReason::EndTurn),
            stop_sequence: None,
            usage: Usage {
                input_tokens: 12,
                output_tokens: 6,
            },
        };

        assert_eq!(response, expected_response);
    }
}
