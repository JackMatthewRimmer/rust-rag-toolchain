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

#[cfg(test)]
mod tests {
    use super::*;
    use mockito::{Mock, Server, ServerGuard};

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

    const ERROR_RESPONSE: &'static str = r#"
    {
        "type": "error",
        "error": {
            "type": "not_found_error",
            "message": "The requested resource could not be found."
        }
    } 
    "#;

    #[tokio::test]
    async fn invoke_correct_response_succeeds() {
        let (client, mut server) = with_mocked_client(None).await;
        let mock = with_mocked_request(&mut server, 200, &CHAT_MESSAGE_RESPONSE);

        let response = client
            .invoke(vec![
                PromptMessage::SystemMessage("You are a comedian".to_string()),
                PromptMessage::HumanMessage("Hello, Claude".to_string()),
            ])
            .await
            .unwrap();

        let expected_response = PromptMessage::AIMessage("Hello!".to_string());
        mock.assert();
        assert_eq!(response, expected_response);
    }

    #[tokio::test]
    async fn invoke_error_response_maps_correctly() {
        let (client, mut server) = with_mocked_client(None).await;
        let mock = with_mocked_request(&mut server, 404, &ERROR_RESPONSE);

        let response = client
            .invoke(vec![
                PromptMessage::SystemMessage("You are a comedian".to_string()),
                PromptMessage::HumanMessage("Hello, Claude".to_string()),
            ])
            .await
            .unwrap_err();

        let expected_reponse =
            AnthropicError::CODE404(serde_json::from_str(ERROR_RESPONSE).unwrap());
        mock.assert();
        assert_eq!(response, expected_reponse);
    }

    // Method which mocks the response the server will give. this
    // allows us to stub the requests instead of sending them to OpenAI
    fn with_mocked_request(
        server: &mut ServerGuard,
        status_code: usize,
        response_body: &str,
    ) -> Mock {
        server
            .mock("POST", "/")
            .with_status(status_code)
            .with_header("Content-Type", "application/json")
            .with_body(response_body)
            .create()
    }

    // This methods returns a client which is pointing at the mocked url
    // and the mock server which we can orchestrate the stubbings on.
    async fn with_mocked_client(
        config: Option<Map<String, Value>>,
    ) -> (AnthropicChatCompletionClient, ServerGuard) {
        std::env::set_var("ANTHROPIC_API_KEY", "fake key");
        let server = Server::new_async().await;
        let url = server.url();
        let model = AnthropicModel::Claude3Point5Sonnet;
        let mut client = match config {
            Some(config) => {
                AnthropicChatCompletionClient::try_new_with_additional_config(model, config)
                    .unwrap()
            }
            None => AnthropicChatCompletionClient::try_new(model).unwrap(),
        };
        client.url = url;
        (client, server)
    }
}
