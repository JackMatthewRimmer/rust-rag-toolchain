use async_trait::async_trait;
use serde_json::{Map, Value};
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
    additional_config: Option<Map<String, Value>>,
}

impl OpenAIChatCompletionClient {
    pub fn try_new(model: OpenAIModel) -> Result<OpenAIChatCompletionClient, VarError> {
        let client: OpenAIHttpClient = OpenAIHttpClient::try_new()?;
        Ok(OpenAIChatCompletionClient {
            url: OPENAI_CHAT_COMPLETIONS_URL.into(),
            client,
            model,
            additional_config: None,
        })
    }

    /// # with_additional_config
    ///
    /// This methods allows users to set additional parameters to be included
    /// in chat completion requests such as top_p, temperature, seed etc. Note
    /// that setting 'n' will not yield anything as we only return the first
    /// message we got back.
    pub fn with_additional_config(&mut self, config: Map<String, Value>) {
        self.additional_config = Some(config);
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
            additional_config: self.additional_config.clone(),
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

#[cfg(test)]
mod chat_completion_client_test {
    use super::*;
    use mockito::{Mock, Server, ServerGuard};

    const CHAT_COMPLETION_RESPONSE: &'static str = r#"
    {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-3.5-turbo-0613",
        "system_fingerprint": "fp_44709d6fcb",
        "choices": [{
          "index": 0,
          "message": {
            "role": "assistant",
            "content": "Hello there, how may I assist you today?"
          },
          "logprobs": null,
          "finish_reason": "stop"
        }],
        "usage": {
          "prompt_tokens": 9,
          "completion_tokens": 12,
          "total_tokens": 21
        }
    }
    "#;

    const ERROR_RESPONSE: &'static str = r#"
    {
        "error": {
            "message": "Incorrect API key provided: fdas. You can find your API key at https://platform.openai.com/account/api-keys.",
            "type": "invalid_request_error",
            "param": null,
            "code": "invalid_api_key"
        }
    }
    "#;

    #[tokio::test]
    async fn test_correct_response_succeeds() {
        let (client, mut server) = with_mocked_client();
        let mock = with_mocked_request(&mut server, 200, CHAT_COMPLETION_RESPONSE);
        let prompt = PromptMessage::HumanMessage("Please ask me a question".into());
        let response = client.invoke(vec![prompt]).await.unwrap();
        let expected_response =
            PromptMessage::AIMessage("Hello there, how may I assist you today?".into());
        mock.assert();
        assert_eq!(expected_response, response);
    }

    #[tokio::test]
    async fn test_error_response_maps_correctly() {
        let (client, mut server) = with_mocked_client();
        let mock = with_mocked_request(&mut server, 401, ERROR_RESPONSE);
        let prompt = PromptMessage::HumanMessage("Please ask me a question".into());
        let response = client.invoke(vec![prompt]).await.unwrap_err();
        let error_body = serde_json::from_str(ERROR_RESPONSE).unwrap();
        let expected_response = OpenAIError::CODE401(error_body);
        mock.assert();
        assert_eq!(expected_response, response);
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
            .with_header("content-type", "application/json")
            .with_body(response_body)
            .create()
    }

    // This methods returns a client which is pointing at the mocked url
    // and the mock server which we can orchestrate the stubbings on.
    fn with_mocked_client() -> (OpenAIChatCompletionClient, ServerGuard) {
        std::env::set_var("OPENAI_API_KEY", "fake key");
        let server = Server::new();
        let url = server.url();
        let model = OpenAIModel::Gpt3Point5;
        let mut client = OpenAIChatCompletionClient::try_new(model).unwrap();
        client.url = url;
        (client, server)
    }
}
