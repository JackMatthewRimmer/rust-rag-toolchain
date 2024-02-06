use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use typed_builder::TypedBuilder;

use crate::clients::types::PromptMessage;

/// See <https://platform.openai.com/docs/api-reference/embeddings/create>
#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, TypedBuilder)]
#[serde(rename_all = "snake_case")]
pub struct ChatCompletionRequest {
    pub model: OpenAIModel,
    pub messages: Vec<ChatMessage>,
    #[builder(default, setter(strip_option))]
    #[serde(flatten)]
    pub additional_config: Option<Map<String, Value>>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: OpenAIModel,
    pub system_fingerprint: String,
    pub choices: Vec<ChatCompletionChoices>,
    pub usage: Usage,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ChatMessageRole {
    System,
    User,
    Assistant,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, Clone, Copy)]
#[serde(rename_all = "snake_case")]
pub enum OpenAIModel {
    #[serde(rename = "gpt-4")]
    Gpt4,
    #[serde(rename = "gpt-3.5")]
    Gpt3Point5,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: ChatMessageRole,
    pub content: String,
}

impl From<PromptMessage> for ChatMessage {
    fn from(prompt_message: PromptMessage) -> Self {
        match prompt_message {
            PromptMessage::SystemMessage(message) => ChatMessage {
                role: ChatMessageRole::System,
                content: message,
            },
            PromptMessage::HumanMessage(message) => ChatMessage {
                role: ChatMessageRole::User,
                content: message,
            },
            PromptMessage::AIMessage(message) => ChatMessage {
                role: ChatMessageRole::Assistant,
                content: message,
            },
        }
    }
}

impl From<ChatMessage> for PromptMessage {
    fn from(value: ChatMessage) -> Self {
        match value.role {
            ChatMessageRole::Assistant => PromptMessage::AIMessage(value.content),
            ChatMessageRole::System => PromptMessage::SystemMessage(value.content),
            ChatMessageRole::User => PromptMessage::HumanMessage(value.content),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChatCompletionChoices {
    pub index: usize,
    pub message: ChatMessage,
    pub logprobs: Option<bool>,
    pub finish_reason: String,
}

mod request_model_tests {

    use super::*;
    const CHAT_COMPLETION_REQUEST: &str = r#"{"model":"gpt-4","messages":[{"role":"system","content":"Hello,howareyou?"},{"role":"user","content":"I'mdoinggreat.Howaboutyou?"},{"role":"system","":"I'mdoingwell.I'mgladtohearyou'redoingwell."}],"temerature":0.7}"#;
    const CHAT_COMPLETION_RESPONSE: &str = r#"{"id":"chatcmpl-123","object":"chat.completion","created":1677652288,"model":"gpt-4","system_fingerprint":"fp_44709d6fcb","choices":[{"index":0,"message":{"role":"assistant","content":"\n\nHello there, how may I assist you today?"},"logprobs":null,"finish_reason":"stop"}],"usage":{"prompt_tokens":9,"completion_tokens":12,"total_tokens":21}}"#;

    #[test]
    fn test_chat_completion_request_serializes() {
        let mut additional_config: Map<String, Value> = Map::new();
        additional_config.insert("temerature".into(), 0.7.into());

        let request: ChatCompletionRequest = ChatCompletionRequest {
            model: OpenAIModel::Gpt4,
            messages: vec![
                ChatMessage {
                    role: ChatMessageRole::System,
                    content: "Hello,howareyou?".into(),
                },
                ChatMessage {
                    role: ChatMessageRole::User,
                    content: "I'mdoinggreat.Howaboutyou?".into(),
                },
                ChatMessage {
                    role: ChatMessageRole::System,
                    content: "I'mdoingwell.I'mgladtohearyou'redoingwell.".into(),
                },
            ],
            additional_config: Some(additional_config),
        };

        let request_json: String = serde_json::to_string(&request).unwrap();
        assert_eq!(request_json, CHAT_COMPLETION_REQUEST);
    }

    #[test]
    fn test_chat_completions_response_deserializes() {
        let response: ChatCompletionResponse =
            serde_json::from_str(CHAT_COMPLETION_RESPONSE).unwrap();

        let expected_response: ChatCompletionResponse = ChatCompletionResponse {
            id: "chatcmpl-123".into(),
            object: "chat.completion".into(),
            created: 1677652288,
            model: OpenAIModel::Gpt4,
            system_fingerprint: "fp_44709d6fcb".into(),
            choices: vec![ChatCompletionChoices {
                index: 0,
                message: ChatMessage {
                    role: ChatMessageRole::Assistant,
                    content: "\n\nHello there, how may I assist you today?".into(),
                },
                logprobs: None,
                finish_reason: "stop".into(),
            }],
            usage: Usage {
                prompt_tokens: 9,
                completion_tokens: 12,
                total_tokens: 21,
            },
        };
        assert_eq!(expected_response, response)
    }
}
