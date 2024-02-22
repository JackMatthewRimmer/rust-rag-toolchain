use crate::{
    chains::ChainError,
    clients::{AsyncChatClient, PromptMessage},
};
use std::iter::once;
use typed_builder::TypedBuilder;

#[derive(Debug, TypedBuilder, Clone, PartialEq, Eq)]
pub struct ChatHistoryChain<T>
where
    T: AsyncChatClient,
{
    chat_history_buffer: ChatHistoryBuffer,
    chat_client: T,
}

impl<T> ChatHistoryChain<T>
where
    T: AsyncChatClient,
{
    /// # new
    ///
    /// This constructor to create a new ChatHistoryChain.
    ///
    /// # Arguments
    /// * `chat_client` - The chat client to be used
    /// * `retriever` - The retriever to be used
    /// * `system_prompt` - The system prompt, please use [`PromptMessage::SystemMessage`]
    pub fn new(chat_client: T, system_prompt: PromptMessage) -> Self {
        let chat_history_buffer = ChatHistoryBuffer::new(system_prompt);
        ChatHistoryChain {
            chat_history_buffer,
            chat_client,
        }
    }

    /// # invoke_chain
    ///
    /// function to execute the ChatHistoryChain given a new user prompt.
    /// Each time this method is invoked, the user message is added to the chat history.
    ///
    /// # Arguments
    /// * `user_message` - the user prompt that will be sent to the LLM along with the chat history.
    ///
    /// # Errors
    /// * [`ChainError::ChatClientError`] if the chat client invocation fails.
    ///
    /// # Returns
    /// * [`PromptMessage::AIMessage`] - the response from the chat client.
    pub async fn invoke_chain(
        &self,
        user_message: PromptMessage,
    ) -> Result<PromptMessage, ChainError<T::ErrorType>> {
        let history: Vec<PromptMessage> = self.chat_history_buffer.get_messages();
        let history_with_prompt = history.into_iter().chain(once(user_message)).collect();
        self.chat_client
            .invoke(history_with_prompt)
            .await
            .map_err(ChainError::ChatClientError)
    }
}

/// # ChatHistoryBuffer
///
/// This struct is used to store the chat history of the conversation.
/// Internal buffer we will hide from the user.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ChatHistoryBuffer {
    messages: Vec<PromptMessage>,
}

impl ChatHistoryBuffer {
    /// # new
    /// we start a chat history buffer with a system prompt.
    fn new(system_prompt: PromptMessage) -> Self {
        ChatHistoryBuffer {
            messages: vec![system_prompt],
        }
    }
    /// # get_messages
    /// return a reference to the messages in the buffer.
    fn get_messages(&self) -> Vec<PromptMessage> {
        self.messages
    }
}
