use crate::{
    chains::ChainError,
    clients::{AsyncChatClient, PromptMessage},
};
use std::cell::RefCell;
use std::iter::once;

#[derive(Debug, Clone, PartialEq, Eq)]
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
    /// # [`ChatHistoryChain::new`]
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

    /// # [`ChatHistoryChain::invoke_chain`]
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
        let history_with_prompt = history
            .into_iter()
            .chain(once(user_message.clone()))
            .collect();
        let response = self
            .chat_client
            .invoke(history_with_prompt)
            .await
            .map_err(ChainError::ChatClientError)?;
        self.chat_history_buffer.append(user_message);
        self.chat_history_buffer.append(response.clone());
        Ok(response)
    }
}

/// # [`ChatHistoryBuffer`]
///
/// This struct is used to store the chat history of the conversation.
/// Internal buffer we will hide from the user.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ChatHistoryBuffer {
    messages: RefCell<Vec<PromptMessage>>,
}

impl ChatHistoryBuffer {
    /// # [`ChatHistoryBuffer::new`]
    /// we start a chat history buffer with a system prompt.
    fn new(system_prompt: PromptMessage) -> Self {
        ChatHistoryBuffer {
            messages: RefCell::new(vec![system_prompt]),
        }
    }
    /// # [`ChatHistoryBuffer::get_messages`]
    /// return a reference to the messages in the buffer.
    fn get_messages(&self) -> Vec<PromptMessage> {
        let reference: &Vec<PromptMessage> = &self.messages.borrow();
        reference.clone()
    }

    /// # [`ChatHistoryBuffer::append`]
    fn append(&self, message: PromptMessage) {
        println!("Appending message: {:?}", message);
        let reference: &mut Vec<PromptMessage> = &mut self.messages.borrow_mut();
        reference.push(message);
    }
}

#[cfg(test)]
mod chat_history_chain_tests {
    use super::*;
    use crate::clients::MockAsyncChatClient;
    use lazy_static::lazy_static;
    use mockall::predicate::eq;
    use std::vec;

    lazy_static! {
        static ref SYSTEM_PROMPT: PromptMessage =
            PromptMessage::SystemMessage("system prompt".into());
        static ref USER_PROMPT_1: PromptMessage = PromptMessage::HumanMessage("user prompt".into());
        static ref USER_PROMPT_2: PromptMessage =
            PromptMessage::HumanMessage("user prompt 2".into());
        static ref AI_RESPONSE: PromptMessage = PromptMessage::AIMessage("AI response".into());
        static ref AI_RESPONSE_2: PromptMessage = PromptMessage::AIMessage("AI response 2".into());
    }

    #[tokio::test]
    async fn test_chat_history_chain() {
        let mut chat_client = MockAsyncChatClient::new();
        chat_client
            .expect_invoke()
            .with(eq(vec![SYSTEM_PROMPT.clone(), USER_PROMPT_1.clone()]))
            .times(1)
            .returning(|_| Ok(AI_RESPONSE.clone()));

        chat_client
            .expect_invoke()
            .with(eq(vec![
                SYSTEM_PROMPT.clone(),
                USER_PROMPT_1.clone(),
                AI_RESPONSE.clone(),
                USER_PROMPT_2.clone(),
            ]))
            .times(1)
            .returning(|_| Ok(AI_RESPONSE_2.clone()));

        let chat_history_chain = ChatHistoryChain::new(chat_client, SYSTEM_PROMPT.clone());
        let result = chat_history_chain
            .invoke_chain(USER_PROMPT_1.clone())
            .await
            .unwrap();
        assert_eq!(result, AI_RESPONSE.clone());

        let result2 = chat_history_chain
            .invoke_chain(USER_PROMPT_2.clone())
            .await
            .unwrap();
        assert_eq!(result2, AI_RESPONSE_2.clone());
    }
}
