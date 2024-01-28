use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use typed_builder::TypedBuilder;

// --------------------------------------------------------------------------------
/// See <https://platform.openai.com/docs/api-reference/embeddings/create>
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
    pub message: String,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, TypedBuilder)]
#[serde(rename_all = "snake_case")]
pub struct ChatCompletionRequest {
    pub model: OpenAIModel,
    pub messages: Vec<ChatMessage>,
    #[builder(default, setter(strip_option))]
    #[serde(flatten)]
    pub additional_config: Option<Map<String, Value>>,
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
                    message: "Hello,howareyou?".into(),
                },
                ChatMessage {
                    role: ChatMessageRole::User,
                    message: "I'mdoinggreat.Howaboutyou?".into(),
                },
                ChatMessage {
                    role: ChatMessageRole::System,
                    message: "I'mdoingwell.I'mgladtohearyou'redoingwell.".into(),
                },
            ],
            additional_config: Some(additional_config)
        };

        let request_json: String = serde_json::to_string(&request).unwrap();
        assert_eq!(request_json, CHAT_COMPLETION_REQUEST);
    }
    
}