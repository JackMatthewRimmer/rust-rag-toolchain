use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use typed_builder::TypedBuilder;

#[derive(Debug, Serialize, Deserialize, PartialEq, TypedBuilder)]
#[serde(rename_all = "snake_case")]
pub struct MessagesRequest {
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub system: String,
    pub model: AnthropicModel,
    pub max_tokens: u32,
    #[builder(default, setter(strip_option))]
    #[serde(flatten)]
    pub additional_config: Option<Map<String, Value>>,
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

/// # [`AnthropicModel`]
///
/// A list of model's available to use in the Anthropic API.
/// note these may have effects on what values are available for config
/// such as max_tokens.
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

    const CHAT_MESSAGE_REQUEST: &str = r#"{"messages":[{"role":"user","content":[{"type":"text","text":"Hello, Claude"}]},{"role":"assistant","content":[{"type":"text","text":"Hello!"}]},{"role":"user","content":[{"type":"text","text":"Can you describe LLMs to me?"}]}],"model":"claude-3-5-sonnet-20240620","max_tokens":1024,"temperature":1}"#;
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
    fn test_serialize_chat_message_request() {
        let mut additional_config: Map<String, Value> = Map::new();
        additional_config.insert("temperature".into(), Value::Number(1.into()));
        let request = MessagesRequest {
            messages: vec![
                Message {
                    role: Role::User,
                    content: vec![Content::Text {
                        text: "Hello, Claude".to_string(),
                    }],
                },
                Message {
                    role: Role::Assistant,
                    content: vec![Content::Text {
                        text: "Hello!".to_string(),
                    }],
                },
                Message {
                    role: Role::User,
                    content: vec![Content::Text {
                        text: "Can you describe LLMs to me?".to_string(),
                    }],
                },
            ],
            system: "".to_string(),
            model: AnthropicModel::Claude3Point5Sonnet,
            max_tokens: 1024,
            additional_config: Some(additional_config),
        };

        let request_json = serde_json::to_string(&request).unwrap();
        assert_eq!(request_json, CHAT_MESSAGE_REQUEST);
    }

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
