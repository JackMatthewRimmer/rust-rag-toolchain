use crate::{
    chains::ChainError,
    clients::{AsyncChatClient, PromptMessage},
};
use std::cell::RefCell;
use std::iter::once;

/// # [`ChatHistoryChain`]
///
/// The chat history chain executes the workflow of saving each message in the conversation history.
/// that is then resent with each subsequent request to the chat client. So previous messages can be
//  referenced in prompts and the LLM will be aware of it.
///
/// * `T` - The type of the chat client to be used
///
/// # Examples
/// ```
///
/// use rag_toolchain::clients::*;
/// use rag_toolchain::chains::*;
///
/// async fn run_chain() {
///     const SYSTEM_MESSAGE: &'static str = "You are a chat bot that must answer questions accurately";
///     let system_prompt = PromptMessage::SystemMessage(SYSTEM_MESSAGE.into());
///     let client = OpenAIChatCompletionClient::try_new(OpenAIModel::Gpt3Point5Turbo).unwrap();
///     let chain = ChatHistoryChain::new(client, system_prompt);
///     let user_prompt1 = PromptMessage::HumanMessage("Please tell me about the weather".into());
///     let response1 = chain.invoke_chain(user_prompt1).await.unwrap();
///     let user_prompt2 =
///     PromptMessage::HumanMessage("What was the last question I just asked ?".into());
///     let response2 = chain.invoke_chain(user_prompt2).await.unwrap();
///     println!("Response 1: {}", response1.content());
///     println!("Response 2: {}", response2.content());
/// }
///
/// ```
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
    /// * `chat_client`: `T` - The chat client to be used
    /// * `system_prompt`: [`PromptMessage`] - The system prompt, please use [`PromptMessage::SystemMessage`]
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
    /// * `user_message`: [`PromptMessage`] - the user prompt that will be sent to the LLM along with the chat history.
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

#[derive(Debug, Clone, PartialEq, Eq)]
struct ChatHistoryBuffer {
    messages: RefCell<Vec<PromptMessage>>,
}

impl ChatHistoryBuffer {
    /// [`ChatHistoryBuffer::new`]
    ///
    /// Creates a new chat history buffer with a system prompt.
    fn new(system_prompt: PromptMessage) -> Self {
        ChatHistoryBuffer {
            messages: RefCell::new(vec![system_prompt]),
        }
    }
    /// # [`ChatHistoryBuffer::get_messages`]
    ///
    /// return a clone of the messages in the buffer.
    fn get_messages(&self) -> Vec<PromptMessage> {
        let reference: &Vec<PromptMessage> = &self.messages.borrow();
        reference.clone()
    }

    /// # [`ChatHistoryBuffer::append`]
    ///
    /// Appends a message to the chat history buffer.
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
