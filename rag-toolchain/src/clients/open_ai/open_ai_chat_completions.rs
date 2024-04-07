use futures::StreamExt;
use reqwest_eventsource::{Event, EventSource};
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

/// # [`OpenAIChatCompletionClient`]
/// Allows for interacting with open ai models
///
/// # Examples
/// ```
/// use rag_toolchain::common::*;
/// use rag_toolchain::clients::*;
/// use serde_json::Map;
/// use serde_json::Value;
/// async fn generate_embedding() {
///     let model: OpenAIModel = OpenAIModel::Gpt3Point5;
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
            url: OPENAI_CHAT_COMPLETIONS_URL.into(),
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
            url: OPENAI_CHAT_COMPLETIONS_URL.into(),
            client,
            model,
            additional_config: Some(additional_config),
        })
    }

    pub async fn streaming_test(&self) {
        let body: ChatCompletionRequest = ChatCompletionRequest {
            model: self.model,
            messages: vec![PromptMessage::HumanMessage("Hello".into()).into()],
            additional_config: self.additional_config.clone(),
        };

        let mut event_source: EventSource = self
            .client
            .send_stream_request(body, &self.url)
            .await
            .unwrap();

        let mut stream = ChatCompletionStream::new(&mut event_source);

        while let Some(value) = stream.next().await {
            println!("{:?}", value);
        }
    }
}

impl AsyncChatClient for OpenAIChatCompletionClient {
    type ErrorType = OpenAIError;

    /// # [`OpenAIChatCompletionClient::invoke`]
    ///
    /// function to execute the ChatCompletion given a list of prompt messages.
    ///
    /// # Arguments
    /// * `prompt_messages` - the list of prompt messages that will be sent to the LLM.
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

pub struct ChatCompletionStream<'a> {
    event_source: &'a mut EventSource,
}

pub enum ChatCompletionStreamValue {
    Connecting,
    /// the response parsed as a prompt message
    Message(PromptMessage),
    /// any errors that occured reading the next value in the stream
    Error(OpenAIError),
    /// This is a signal that the stream has ended.
    EOS,
}

impl<'a> ChatCompletionStream<'a> {
    const STOP_MESSAGE: &'static str = "[DONE]";

    /// # [`ChatCompletionStream::new`]
    ///
    /// This struct just wraps the EventSource when from the
    /// context of streaming chat completions.
    pub fn new(event_source: &'a mut EventSource) -> Self {
        Self { event_source }
    }

    pub async fn next(&mut self) -> Option<String> {
        self.event_source.next().await.map(|event| match event {
            Ok(Event::Open) => "Connecting".into(),
            Ok(Event::Message(msg)) => {
                if msg.data == Self::STOP_MESSAGE {
                    self.event_source.close();
                    "EOS".into()
                } else {
                    msg.data
                }
            }
            Err(e) => {
                self.event_source.close();
                e.to_string()
            }
        })

        // let value: ChatCompletionStreamValue = match self.event_source.next().await {
        //     Some(event) => match event {
        //         Ok(Event::Open) => ChatCompletionStreamValue::Connecting,
        //         Ok(Event::Message(msg)) => {
        //             println!("{:?}", msg.data);
        //             if msg.data == Self::STOP_MESSAGE {
        //                 ChatCompletionStreamValue::EOS
        //             } else {
        //                 Self::parse_message(&msg.data)
        //             }
        //         }
        //         Err(e) => {
        //             ChatCompletionStreamValue::Error(OpenAIError::ErrorReadingStream(e.to_string()))
        //         }
        //     },
        //     None => ChatCompletionStreamValue::Connecting,
        // };

        // self.event_source.close();
        // value
    }

    /// # [`ChatCompletionStream::parse_message`]
    ///
    /// Helper method to deserialize the message from the event source.
    ///
    /// # Arguments
    /// * `msg`: &[`str`] - the raw response from the event source.
    fn parse_message(msg: &str) -> ChatCompletionStreamValue {
        match serde_json::from_str::<ChatCompletionResponse>(msg) {
            Ok(deserialized_message) => {
                return ChatCompletionStreamValue::Message(
                    deserialized_message
                        .choices
                        .get(0)
                        .unwrap()
                        .message
                        .clone()
                        .into(),
                );
            }
            Err(e) => {
                return ChatCompletionStreamValue::Error(
                    OpenAIError::ErrorDeserializingResponseBody(200, e.to_string()),
                );
            }
        }
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
    async fn streaming_test() {
        let mut additional_config: Map<String, Value> = Map::new();
        additional_config.insert("stream".into(), true.into());
        let client: OpenAIChatCompletionClient =
            OpenAIChatCompletionClient::try_new_with_additional_config(
                OpenAIModel::Gpt3Point5,
                additional_config,
            )
            .unwrap();
        client.streaming_test().await;
    }

    #[tokio::test]
    async fn test_correct_response_succeeds() {
        let (client, mut server) = with_mocked_client().await;
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
        let (client, mut server) = with_mocked_client().await;
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
    async fn with_mocked_client() -> (OpenAIChatCompletionClient, ServerGuard) {
        std::env::set_var("OPENAI_API_KEY", "fake key");
        let server = Server::new_async().await;
        let url = server.url();
        let model = OpenAIModel::Gpt3Point5;
        let mut client = OpenAIChatCompletionClient::try_new(model).unwrap();
        client.url = url;
        (client, server)
    }
}
