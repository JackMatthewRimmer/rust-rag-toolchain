use async_trait::async_trait;
use std::env::VarError;

use crate::clients::open_ai::model::chat_completions::{
    ChatCompletionChoices, ChatCompletionRequest, ChatCompletionResponse, OpenAIModel,
};
use crate::clients::open_ai::open_ai_core::OpenAIHttpClient;
use crate::clients::{AsyncChatClient, PromptMessage};

use super::model::chat_completions::ChatMessage;
use super::model::errors::OpenAIError;

const OPENAI_CHAT_COMPLETIONS_URL: &str = "https://api.openai.com/v1/chat/completions";

/// # OpenAIChatCompletionClient
/// Allows for interacting with open ai models
///
/// # Required Environment Variables
/// OPENAI_API_KEY: The API key to use for the OpenAI API
pub struct OpenAIChatCompletionClient {
    url: String,
    client: OpenAIHttpClient,
    model: OpenAIModel,
}

impl OpenAIChatCompletionClient {
    pub fn try_new(model: OpenAIModel) -> Result<OpenAIChatCompletionClient, VarError> {
        let client: OpenAIHttpClient = OpenAIHttpClient::try_new()?;
        Ok(OpenAIChatCompletionClient {
            url: OPENAI_CHAT_COMPLETIONS_URL.into(),
            client,
            model,
        })
    }
}

#[async_trait]
impl AsyncChatClient for OpenAIChatCompletionClient {
    type ErrorType = OpenAIError;

    async fn invoke(
        &self,
        prompt_messages: Vec<PromptMessage>,
    ) -> Result<PromptMessage, Self::ErrorType> {
        let mapped_messages: Vec<ChatMessage> =
            prompt_messages.into_iter().map(ChatMessage::from).collect();

        let body: ChatCompletionRequest = ChatCompletionRequest {
            model: self.model,
            messages: mapped_messages,
            additional_config: None,
        };

        let response: ChatCompletionResponse = self.client.send_request(body, &self.url).await?;
        let choices: Vec<ChatCompletionChoices> = response.choices;
        let messages: Vec<PromptMessage> = choices
            .into_iter()
            .map(|x| PromptMessage::from(x.message))
            .collect();

        Ok(messages[0].clone())
    }
}
