use futures::StreamExt;
use reqwest_eventsource::{Event, EventSource};
use serde_json::{Map, Value};
use std::env::VarError;

use crate::clients::open_ai::model::chat_completions::{
    ChatCompletionChoices, ChatCompletionRequest, ChatCompletionResponse, OpenAIModel,
};
use crate::clients::open_ai::open_ai_core::OpenAIHttpClient;
use crate::clients::{
    AsyncChatClient, AsyncStreamedChatClient, ChatCompletionStream, PromptMessage,
};

use super::model::chat_completions::{
    ChatCompletionDelta, ChatCompletionStreamedResponse, ChatMessage,
};

use super::model::errors::OpenAIError;

/// # [`OpenAIChatCompletionClient`]
/// Allows for interacting with open ai models
/// # Examples
/// ```
/// use rag_toolchain::common::*;
/// use rag_toolchain::clients::*;
/// use serde_json::Map;
/// use serde_json::Value;
/// async fn generate_completion() {
///     let model: OpenAIModel = OpenAIModel::Gpt3Point5Turbo;
///     let mut additional_config: Map<String, Value> = Map::new();
///     additional_config.insert("temperature".into(), 0.5.into());
///
///     let client: OpenAIChatCompletionClient =
///         OpenAIChatCompletionClient::try_new_with_additional_config(model, additional_config).unwrap();
///     let system_message: PromptMessage = PromptMessage::SystemMessage(
///         "You are a comedian that cant ever reply to someone unless its phrased as a sarcastic joke"
///         .into());
///     let user_message: PromptMessage =
///     PromptMessage::HumanMessage("What is the weather like today ?".into());
///     let reply = client
///         .invoke(vec![system_message, user_message])
///         .await
///         .unwrap();
///     println!("{:?}", reply.content());
/// }
/// ```
/// # Required Environment Variables
/// OPENAI_API_KEY: The API key to use for the OpenAI API
pub struct OpenAIChatCompletionClient {
    url: String,
    client: OpenAIHttpClient,
    model: OpenAIModel,
    additional_config: Option<Map<String, Value>>,
}

impl OpenAIChatCompletionClient {
    const OPENAI_CHAT_COMPLETIONS_URL: &str = "https://api.openai.com/v1/chat/completions";

    /// # [`OpenAIChatCompletionClient::try_new`]
    ///
    /// This method creates a new OpenAIChatCompletionClient. All inference parameters used
    /// will be the default ones provided by the OpenAI API.
    ///
    /// # Arguments
    /// * `model`: [`OpenAIModel`] - The model to use for the chat completion.
    ///
    /// # Errors
    /// * [`VarError`] - if the OPENAI_API_KEY environment variable is not set.
    ///
    /// # Returns
    /// * [`OpenAIChatCompletionClient`] - the chat completion client.
    pub fn try_new(model: OpenAIModel) -> Result<OpenAIChatCompletionClient, VarError> {
        let client: OpenAIHttpClient = OpenAIHttpClient::try_new()?;
        Ok(OpenAIChatCompletionClient {
            url: Self::OPENAI_CHAT_COMPLETIONS_URL.into(),
            client,
            model,
            additional_config: None,
        })
    }

    /// # [`OpenAIChatCompletionClient::try_new_with_additional_config`]
    ///
    /// This method creates a new OpenAIChatCompletionClient. All inference parameters provided
    /// in the additional_config will be used in the chat completion request. an example of this
    /// could be 'temperature', 'top_p', 'seed' etc.
    ///
    /// # Forbidden Properties
    /// * "stream": this cannot be set as it is used internally by the client.
    /// * "n": n can be set but will result in wasted tokens as the client is built for single
    ///        chat completions. We intend to add support for multiple completions in the future.
    ///
    /// # Arguments
    /// * `model`: [`OpenAIModel`] - The model to use for the chat completion.
    /// * `additional_config`: [`Map<String, Value>`] - The additional configuration to use for the chat completion.
    ///
    /// # Errors
    /// * [`VarError`] - if the OPENAI_API_KEY environment variable is not set.
    ///
    /// # Returns
    /// * [`OpenAIChatCompletionClient`] - the chat completion client.
    pub fn try_new_with_additional_config(
        model: OpenAIModel,
        additional_config: Map<String, Value>,
    ) -> Result<OpenAIChatCompletionClient, VarError> {
        let client: OpenAIHttpClient = OpenAIHttpClient::try_new()?;
        Ok(OpenAIChatCompletionClient {
            url: Self::OPENAI_CHAT_COMPLETIONS_URL.into(),
            client,
            model,
            additional_config: Some(additional_config),
        })
    }

    /// # [`OpenAIChatCompletionClient::try_new_with_url`]
    ///
    /// This method creates a new OpenAIChatCompletionClient. All inference parameters used
    /// will be the default ones provided by the OpenAI API. You can pass the url in directly
    /// # Arguments
    /// * `model`: [`OpenAIModel`] - The model to use for the chat completion.
    /// * `url`: [`String`] - The url to use for the api call.
    ///
    /// # Errors
    /// * [`VarError`] - if the OPENAI_API_KEY environment variable is not set.
    ///
    /// # Returns
    /// * [`OpenAIChatCompletionClient`] - the chat completion client.
    pub fn try_new_with_url(
        model: OpenAIModel,
        url: String,
    ) -> Result<OpenAIChatCompletionClient, VarError> {
        let client: OpenAIHttpClient = OpenAIHttpClient::try_new()?;
        Ok(OpenAIChatCompletionClient {
            url,
            client,
            model,
            additional_config: None,
        })
    }

    /// # [`OpenAIChatCompletionClient::try_new_with_url_and_additional_config`]
    ///
    /// This method creates a new OpenAIChatCompletionClient. All inference parameters provided
    /// in the additional_config will be used in the chat completion request. an example of this
    /// could be 'temperature', 'top_p', 'seed' etc. You can pass the url in directly.
    ///
    /// # Forbidden Properties
    /// * "stream": this cannot be set as it is used internally by the client.
    /// * "n": n can be set but will result in wasted tokens as the client is built for single
    ///        chat completions. We intend to add support for multiple completions in the future.
    ///
    /// # Arguments
    /// * `model`: [`OpenAIModel`] - The model to use for the chat completion.
    /// * `url`: [`String`] - The url to use for the api call.
    /// * `additional_config`: [`Map<String, Value>`] - The additional configuration to use for the chat completion.
    ///
    /// # Errors
    /// * [`VarError`] - if the OPENAI_API_KEY environment variable is not set.
    ///
    /// # Returns
    /// * [`OpenAIChatCompletionClient`] - the chat completion client.
    pub fn try_new_with_url_and_additional_config(
        model: OpenAIModel,
        url: String,
        additional_config: Map<String, Value>,
    ) -> Result<OpenAIChatCompletionClient, VarError> {
        let client: OpenAIHttpClient = OpenAIHttpClient::try_new()?;
        Ok(OpenAIChatCompletionClient {
            url,
            client,
            model,
            additional_config: Some(additional_config),
        })
    }
}

impl AsyncChatClient for OpenAIChatCompletionClient {
    type ErrorType = OpenAIError;

    /// # [`OpenAIChatCompletionClient::invoke`]
    ///
    /// function to execute the ChatCompletion given a list of prompt messages.
    ///
    /// # Arguments
    /// * `prompt_messages`: [`Vec<PromptMessage>`] - the list of prompt messages that will be sent to the LLM.
    ///
    /// # Errors
    /// * [`OpenAIError`] - if the chat client invocation fails.
    ///
    /// # Returns
    /// [`PromptMessage::AIMessage`] - the response from the chat client.
    async fn invoke(
        &self,
        prompt_messages: Vec<PromptMessage>,
    ) -> Result<PromptMessage, Self::ErrorType> {
        let mapped_messages: Vec<ChatMessage> =
            prompt_messages.into_iter().map(ChatMessage::from).collect();

        let body: ChatCompletionRequest = ChatCompletionRequest {
            model: self.model,
            messages: mapped_messages,
            stream: false,
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

impl AsyncStreamedChatClient for OpenAIChatCompletionClient {
    type ErrorType = OpenAIError;
    type Item = OpenAICompletionStream;

    /// # [`OpenAIChatCompletionClient::invoke_stream`]
    ///
    /// function to execute the ChatCompletion given a list of prompt messages.
    ///
    /// # Arguments
    /// * `prompt_messages`: [`PromptMessage`] - the list of prompt messages that will be sent to the LLM.
    ///
    /// # Errors
    /// * [`OpenAIError`] - if the chat client invocation fails.
    ///
    /// # Returns
    /// impl [`ChatCompletionStream`] - the response from the chat client.
    async fn invoke_stream(
        &self,
        prompt_messages: Vec<PromptMessage>,
    ) -> Result<Self::Item, Self::ErrorType> {
        let mapped_messages: Vec<ChatMessage> =
            prompt_messages.into_iter().map(ChatMessage::from).collect();

        let body: ChatCompletionRequest = ChatCompletionRequest {
            model: self.model,
            messages: mapped_messages,
            stream: true,
            additional_config: self.additional_config.clone(),
        };

        let event_source: EventSource = self.client.send_stream_request(body, &self.url).await?;
        Ok(OpenAICompletionStream::new(event_source))
    }
}

/// [`OpenAICompletionStream`]
///
/// This structs wraps the EventSource and parses returned
/// messages into prompt messages on demand.
pub struct OpenAICompletionStream {
    event_source: EventSource,
}

/// [`CompletionStreamValue`]
///
/// Value returned from each iteration of the stream.
/// Given we wanted to represent connecting as a non-failure
/// state we had to create a new enum to represent this.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompletionStreamValue {
    Connecting,
    Message(PromptMessage),
}

impl OpenAICompletionStream {
    const STOP_MESSAGE: &'static str = "[DONE]";

    /// # [`OpenAICompletionStream::new`]
    ///
    /// This struct just wraps the EventSource when from the
    /// context of streaming chat completions.
    pub fn new(event_source: EventSource) -> Self {
        Self { event_source }
    }

    /// # [`ChatCompletionStream::parse_message`]
    ///
    /// Helper method to deserialize the raw response message from the event source.
    ///
    /// # Arguments
    /// * `msg`: &[`str`] - the raw response from the event source.
    ///
    /// # Errors
    /// * [`OpenAIError`] - if the chat client invocation fails.
    ///
    /// # Returns
    /// * [`Option<Result<CompletionStreamValue, OpenAIError>`] - the response from the chat client.
    ///         None represents the stream is finished. and Some(Err) represents an error.
    ///
    fn parse_message(msg: &str) -> Option<Result<CompletionStreamValue, OpenAIError>> {
        let response: ChatCompletionStreamedResponse = match serde_json::from_str(msg) {
            Ok(response) => response,
            Err(e) => {
                return Some(Err(OpenAIError::ErrorDeserializingResponseBody(
                    200,
                    e.to_string(),
                )));
            }
        };
        let chat_message: ChatCompletionDelta = response.choices.first().unwrap().delta.clone();
        match chat_message.content {
            Some(msg) => {
                let prompt_message: PromptMessage = PromptMessage::AIMessage(msg);
                Some(Ok(CompletionStreamValue::Message(prompt_message)))
            }
            None => None,
        }
    }
}

impl ChatCompletionStream for OpenAICompletionStream {
    type ErrorType = OpenAIError;
    type Item = CompletionStreamValue;

    /// # [`ChatCompletionStream::next`]
    ///
    /// Method to iterate over the completion stream. Note it blocks
    /// until the next message is received. At which point it will
    /// parse the response into a [`CompletionStreamValue`].
    ///
    /// # Examples
    /// ```
    /// use rag_toolchain::clients::*;
    ///
    /// async fn stream_chat_completions(client: OpenAIChatCompletionClient) {
    ///     let user_message: PromptMessage = PromptMessage::HumanMessage("Please ask me a question".into());
    ///     let mut stream: OpenAICompletionStream = client.invoke_stream(vec![user_message]).await.unwrap();
    ///     while let Some(response) = stream.next().await {
    ///         match response {
    ///            Ok(CompletionStreamValue::Connecting) => {},
    ///            Ok(CompletionStreamValue::Message(msg)) => {
    ///                 println!("{:?}", msg.content());
    ///            }
    ///            Err(e) => {
    ///                 println!("{:?}", e);
    ///                 break;
    ///            }
    ///         }
    ///     }
    /// }
    /// ```
    ///
    /// # Errors
    /// * [`OpenAIError::ErrorReadingStream`] - if there was an error reading from the stream.
    /// * [`OpenAIError::ErrorDeserializingResponseBody`] - if there was an error deserializing the response body.
    ///
    /// # Returns
    /// * [`Option<Result<CompletionStreamValue, OpenAIError>>`] - the response from the chat client.
    ///         None represents the stream is finished..
    async fn next(&mut self) -> Option<Result<Self::Item, Self::ErrorType>> {
        let event: Result<Event, reqwest_eventsource::Error> = self.event_source.next().await?;

        let event: Event = match event {
            Ok(event) => event,
            Err(e) => {
                self.event_source.close();
                return Some(Err(OpenAIError::ErrorReadingStream(e.to_string())));
            }
        };

        match event {
            Event::Message(msg) => {
                if msg.data == Self::STOP_MESSAGE {
                    self.event_source.close();
                    None
                } else {
                    Self::parse_message(&msg.data)
                }
            }
            Event::Open => Some(Ok(CompletionStreamValue::Connecting)),
        }
    }
}

#[cfg(test)]
mod tests {
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

    const STREAMED_CHAT_COMPLETION_RESPONSE: &'static str = "id:1\ndata:{\"id\":\"chatcmpl-9BRO0Nnca1ZtfMkFc5tOpQNSJ2Eo0\",\"object\":\"chat.completion.chunk\",\"created\":1712513908,\"model\":\"gpt-3.5-turbo-0125\",\"system_fingerprint\":\"fp_b28b39ffa8\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hello\"},\"logprobs\":null,\"finish_reason\":null}]}\n\ndata:[DONE]\n\n";

    #[test]
    fn try_new_with_env_var_succeeds() {
        std::env::set_var("OPENAI_API_KEY", "test");
        OpenAIChatCompletionClient::try_new(OpenAIModel::Gpt4o)
            .expect("Failed to create OpenAIChatCompletionClient");
    }

    #[test]
    fn try_new_with_additional_config_succeeds() {
        OpenAIChatCompletionClient::try_new_with_additional_config(OpenAIModel::Gpt4o, Map::new())
            .expect("Failed to create OpenAIChatCompletionClient");
    }

    #[tokio::test]
    async fn invoke_correct_response_succeeds() {
        let (client, mut server) = with_mocked_client(None).await;
        let mock = with_mocked_request(&mut server, 200, CHAT_COMPLETION_RESPONSE);
        let prompt = PromptMessage::HumanMessage("Please ask me a question".into());
        let response = client.invoke(vec![prompt]).await.unwrap();
        let expected_response =
            PromptMessage::AIMessage("Hello there, how may I assist you today?".into());
        mock.assert();
        assert_eq!(expected_response, response);
    }

    #[tokio::test]
    async fn invoke_error_response_maps_correctly() {
        let (client, mut server) = with_mocked_client(Some(Map::new())).await;
        let mock = with_mocked_request(&mut server, 401, ERROR_RESPONSE);
        let prompt = PromptMessage::HumanMessage("Please ask me a question".into());
        let response = client.invoke(vec![prompt]).await.unwrap_err();
        let error_body = serde_json::from_str(ERROR_RESPONSE).unwrap();
        let expected_response = OpenAIError::CODE401(error_body);
        mock.assert();
        assert_eq!(expected_response, response);
    }

    #[tokio::test]
    async fn invoke_stream_correct_response_succeeds() {
        let (client, mut server) = with_mocked_client(None).await;
        let mock = server
            .mock("POST", "/")
            .with_status(200)
            .with_header("Content-Type", "text/event-stream")
            .with_body(STREAMED_CHAT_COMPLETION_RESPONSE)
            .create();
        let prompt = PromptMessage::HumanMessage("Please ask me a question".into());
        let mut stream = client.invoke_stream(vec![prompt]).await.unwrap();
        let value1 = stream.next().await.unwrap().unwrap();
        assert_eq!(value1, CompletionStreamValue::Connecting);
        let value2 = stream.next().await.unwrap().unwrap();
        assert_eq!(
            value2,
            CompletionStreamValue::Message(PromptMessage::AIMessage("Hello".into()))
        );
        let value3 = stream.next().await;
        assert_eq!(value3, None);
        mock.assert();
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
    ) -> (OpenAIChatCompletionClient, ServerGuard) {
        std::env::set_var("OPENAI_API_KEY", "fake key");
        let server = Server::new_async().await;
        let url = server.url();
        let model = OpenAIModel::Gpt3Point5Turbo;
        let client = match config {
            Some(config) => OpenAIChatCompletionClient::try_new_with_url_and_additional_config(
                model, url, config,
            )
            .unwrap(),
            None => OpenAIChatCompletionClient::try_new_with_url(model, url).unwrap(),
        };
        (client, server)
    }
}
