use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use typed_builder::TypedBuilder;

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

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
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

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChatCompletionChoices {
    pub message: ChatCompletionChoice,
    pub logprobs: Option<bool>,
    pub finish_reason: String,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChatCompletionChoice {
    pub message: ChatMessage,
    pub index: usize,
}

pub mod request_model_tests {

    use super::*;
    const CHAT_COMPLETION_REQUEST: &str = r#"{"model":"gpt-4","messages":[{"role":"system","message":"Hello,howareyou?"},{"role":"user","message":"I'mdoinggreat.Howaboutyou?"},{"role":"system","message":"I'mdoingwell.I'mgladtohearyou'redoingwell."}],"temerature":0.7}"#;

    #[test]
    fn test_chat_completion_request() {
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
            additional_config: Some(additional_config)
        };

        let request_json: String = serde_json::to_string(&request).unwrap();
        assert_eq!(request_json, CHAT_COMPLETION_REQUEST);
    }
    
}