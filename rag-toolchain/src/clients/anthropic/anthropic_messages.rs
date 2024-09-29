use std::env::VarError;

use crate::clients::anthropic::model::chat_completions::{
    Content, MessagesRequest, MessagesResponse,
};
use crate::clients::{AsyncChatClient, PromptMessage};

use super::model::chat_completions::{AnthropicModel, Message, Role};
use super::{anthropic_core::AnthropicHttpClient, model::errors::AnthropicError};

use serde_json::{Map, Value};

const ANTHROPIC_MESSAGES_URL: &str = "https://api.anthropic.com/v1/messages";

/// # [`AnthropicChatCompletionClient`]
/// Allows for interacting with the Anthropic models via the messages API.
///
/// # Examples
/// ```
/// use serde_json::{Map, Value};
/// use rag_toolchain::clients::*;
/// use rag_toolchain::common::*;
/// async fn generate_completion() {
///     let model: AnthropicModel = AnthropicModel::Claude3Sonnet;
///     let mut additional_config: Map<String, Value> = Map::new();
///     additional_config.insert("temperature".into(), 0.5.into());
///
///     let client: AnthropicChatCompletionClient =
///         AnthropicChatCompletionClient::try_new_with_additional_config(
///             model,
///             4096,
///             additional_config,
///         )
///         .unwrap();
///
///     let system_message: PromptMessage =
///         PromptMessage::SystemMessage("You only reply in a bullet point list".into());
///     let user_message: PromptMessage = PromptMessage::HumanMessage("How does the water flow".into());
///
///     // We invoke the chat client with a list of messages
///     let reply = client
///         .invoke(vec![system_message.clone(), user_message.clone()])
///         .await
///         .unwrap();
///     println!("{:?}", reply.content());
/// }
/// ```
/// # Required Environment Variables
/// ANTHROPIC_API_KEY: The API key for the Anthropic API
pub struct AnthropicChatCompletionClient {
    url: String,
    client: AnthropicHttpClient,
    model: AnthropicModel,
    additional_config: Option<Map<String, Value>>,
    /// This is a required field on the messages API.
    /// Please refer to the API documentation for more information.
    max_tokens: u32,
}

impl AnthropicChatCompletionClient {
    const SYSTEM_MESSAGE_ERROR: &'static str = r#"
        System prompts should be included within the system field of the request.
        This error means that it was attempted to be included in the messages field.
    "#;
    /// # [`AnthropicChatCompletionClient::try_new`]
    ///
    /// This method creates a new instance of the AnthropicChatCompletionClient. All optional
    /// inference parameters will be set to their default values on Anthropic's end.
    ///
    /// # Arguments
    /// * `model`: [`AnthropicModel`] - The model to use for the chat completion.
    /// * `max_tokens`: [`u32`] - The maximum number of tokens to generate in the response.
    ///                           See the API documentation for more information.
    ///
    /// # Errors
    /// [`VarError`] - This error is returned when the ANTHROPIC_API_KEY environment variable is not set.
    ///
    /// # Returns
    /// [`AnthropicChatCompletionClient`] - The client to interact with the Anthropic API.
    pub fn try_new(model: AnthropicModel, max_tokens: u32) -> Result<Self, VarError> {
        let client: AnthropicHttpClient = AnthropicHttpClient::try_new()?;
        Ok(AnthropicChatCompletionClient {
            url: ANTHROPIC_MESSAGES_URL.to_string(),
            client,
            model,
            additional_config: None,
            max_tokens,
        })
    }

    /// # [`AnthropicChatCompletionClient::try_new_with_additional_config`]
    ///
    /// This method creates a new instance of the AnthropicChatCompletionClient. All optional
    /// inference parameters will be set to their default values on Anthropic's end.
    ///
    /// # Arguments
    /// * `model`: [`AnthropicModel`] - The model to use for the chat completion.
    /// * `max_tokens`: [`u32`] - The maximum number of tokens to generate in the response.
    ///                           See the API documentation for more information.
    /// * `additional_config`: [`Map<String, Value>`] - Additional configuration to pass to the API.
    ///                        See the API documentation for more information.
    ///                        Examples of this can be temperature, top_p, etc.  
    /// # Errors
    /// [`VarError`] - This error is returned when the ANTHROPIC_API_KEY environment variable is not set.
    ///
    /// # Returns
    /// [`AnthropicChatCompletionClient`] - The client to interact with the Anthropic API.
    pub fn try_new_with_additional_config(
        model: AnthropicModel,
        max_tokens: u32,
        additional_config: Map<String, Value>,
    ) -> Result<Self, VarError> {
        let client: AnthropicHttpClient = AnthropicHttpClient::try_new()?;
        Ok(AnthropicChatCompletionClient {
            url: ANTHROPIC_MESSAGES_URL.to_string(),
            client,
            model,
            additional_config: Some(additional_config),
            max_tokens,
        })
    }

    /// # [`AnthropicChatCompletionClient::map_prompt_message_to_anthropic_message`]
    ///
    /// Helper method to map the prompt message to the Anthropic message. We work on
    /// the assumption that the system messages were already pulled out of the list of
    /// [`PromptMessage`] therefore if any are passed to this function with throw an error.
    ///
    /// # Arguments
    /// * `prompt_message`: [`PromptMessage`] - The message to map to the Anthropic message.
    ///
    /// # Errors
    /// [`AnthropicError`] - This error is returned when a system message is passed to the function.
    ///
    /// # Returns
    /// [`Message`] - The message to send to the Anthropic API.
    fn map_prompt_message_to_anthropic_message(
        prompt_message: PromptMessage,
    ) -> Result<Message, AnthropicError> {
        match prompt_message {
            PromptMessage::SystemMessage(_) => Err(AnthropicError::Undefined(
                0,
                AnthropicChatCompletionClient::SYSTEM_MESSAGE_ERROR.to_string(),
            )),
            PromptMessage::AIMessage(message) => Ok(Message {
                role: Role::Assistant,
                content: vec![Content::Text { text: message }],
            }),
            PromptMessage::HumanMessage(message) => Ok(Message {
                role: Role::User,
                content: vec![Content::Text { text: message }],
            }),
        }
    }
}

impl AsyncChatClient for AnthropicChatCompletionClient {
    type ErrorType = AnthropicError;

    /// # [`AnthropicChatCompletionClient::invoke`]
    ///
    /// Function to send a list of [`PromptMessage`] to the Anthropic API and receive a response.
    ///
    /// # Arguments
    /// * `prompt_messages`: [`Vec<PromptMessage>`] - The list of messages to send to the API.
    ///
    /// # Errors
    /// * [`AnthropicError`] - This error is returned when the API returns an error.
    ///
    /// # Returns
    /// [`PromptMessage::AIMessage`] - The response from the API.
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
            max_tokens: self.max_tokens,
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
        let additonal_config = Map::new();
        let (client, mut server) = with_mocked_client(Some(additonal_config)).await;
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

    #[test]
    fn map_prompt_message_to_anthropic_message_with_system_message_returns_error() {
        let system_message = PromptMessage::SystemMessage("Hello".to_string());
        let response =
            AnthropicChatCompletionClient::map_prompt_message_to_anthropic_message(system_message)
                .unwrap_err();
        let expected_response = AnthropicError::Undefined(
            0,
            AnthropicChatCompletionClient::SYSTEM_MESSAGE_ERROR.to_string(),
        );
        assert_eq!(response, expected_response);
    }

    #[test]
    fn map_prompt_message_to_anthropic_message_with_human_message_returns_message() {
        let human_message = PromptMessage::HumanMessage("Hello".to_string());
        let response =
            AnthropicChatCompletionClient::map_prompt_message_to_anthropic_message(human_message)
                .unwrap();
        let expected_response = Message {
            role: Role::User,
            content: vec![Content::Text {
                text: "Hello".to_string(),
            }],
        };
        assert_eq!(response, expected_response);
    }

    #[test]
    fn map_prompt_message_to_anthropic_message_with_ai_message_returns_message() {
        let ai_message = PromptMessage::AIMessage("Hello".to_string());
        let response =
            AnthropicChatCompletionClient::map_prompt_message_to_anthropic_message(ai_message)
                .unwrap();
        let expected_response = Message {
            role: Role::Assistant,
            content: vec![Content::Text {
                text: "Hello".to_string(),
            }],
        };
        assert_eq!(response, expected_response);
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
                AnthropicChatCompletionClient::try_new_with_additional_config(model, 1024, config)
                    .unwrap()
            }
            None => AnthropicChatCompletionClient::try_new(model, 1024).unwrap(),
        };
        client.url = url;
        (client, server)
    }
}
