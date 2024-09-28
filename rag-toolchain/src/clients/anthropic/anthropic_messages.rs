use std::env::VarError;

use crate::clients::anthropic::model::chat_completions::{
    Content, MessagesRequest, MessagesResponse,
};
use crate::clients::{AsyncChatClient, PromptMessage};

use super::model::chat_completions::{AnthropicModel, Message, Role};
use super::{anthropic_core::AnthropicHttpClient, model::errors::AnthropicError};

use serde_json::{Map, Value};

const ANTHROPIC_MESSAGES_URL: &str = "https://api.anthropic.com/v1/messages";

pub struct AnthropicChatCompletionClient {
    url: String,
    client: AnthropicHttpClient,
    model: AnthropicModel,
    additional_config: Option<Map<String, Value>>,
}

impl AnthropicChatCompletionClient {
    pub fn try_new(model: AnthropicModel) -> Result<Self, VarError> {
        let client: AnthropicHttpClient = AnthropicHttpClient::try_new()?;
        Ok(AnthropicChatCompletionClient {
            url: ANTHROPIC_MESSAGES_URL.to_string(),
            client,
            model,
            additional_config: None,
        })
    }

    pub fn try_new_with_additional_config(
        model: AnthropicModel,
        additional_config: Map<String, Value>,
    ) -> Result<Self, VarError> {
        let client: AnthropicHttpClient = AnthropicHttpClient::try_new()?;
        Ok(AnthropicChatCompletionClient {
            url: ANTHROPIC_MESSAGES_URL.to_string(),
            client,
            model,
            additional_config: Some(additional_config),
        })
    }

    fn map_prompt_message_to_anthropic_message(
        prompt_message: PromptMessage,
    ) -> Result<Message, AnthropicError> {
        let system_error_message = r#"
            System prompts should be included within the system field of the request.
            This error means that it was attempted to be included in the messages field. 
        "#;

        match prompt_message {
            PromptMessage::SystemMessage(_) => Err(AnthropicError::Undefined(
                0,
                system_error_message.to_string(),
            )),
            PromptMessage::AIMessage(message) => Ok(Message {
                role: Role::Assistant,
                content: vec![Content::Text {
                    text: message.into(),
                }],
            }),
            PromptMessage::HumanMessage(message) => Ok(Message {
                role: Role::User,
                content: vec![Content::Text {
                    text: message.into(),
                }],
            }),
        }
    }
}

impl AsyncChatClient for AnthropicChatCompletionClient {
    type ErrorType = AnthropicError;

    async fn invoke(
        &self,
        prompt_messages: Vec<PromptMessage>,
    ) -> Result<PromptMessage, Self::ErrorType> {
        let mut system_message_content = String::new();
        let mut anthropic_messages = Vec::new();

        for prompt_message in prompt_messages {
            match prompt_message {
                PromptMessage::SystemMessage(content) => {
                    system_message_content.push_str(&content);
                    system_message_content.push('\n');
                }
                _ => {
                    let anthropic_message =
                        Self::map_prompt_message_to_anthropic_message(prompt_message)?;
                    anthropic_messages.push(anthropic_message);
                }
            }
        }

        let request: MessagesRequest = MessagesRequest {
            messages: anthropic_messages,
            system: system_message_content,
            model: self.model.clone(),
            additional_config: self.additional_config.clone(),
        };

        let response: MessagesResponse = self.client.send_request(request, &self.url).await?;
        let response_message = match response.content[0] {
            Content::Text { ref text } => PromptMessage::AIMessage(text.clone()),
        };

        Ok(response_message)
    }
}
