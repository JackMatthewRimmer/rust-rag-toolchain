use crate::{
    chains::{utils::build_prompt, RagChainError},
    clients::{AsyncChatClient, PromptMessage},
    common::Chunk,
    retrievers::AsyncRetriever,
};
use std::num::NonZeroU32;
use typed_builder::TypedBuilder;

/// # BasicRAGChain
///
/// This struct allows for easily executing RAG given a single user prompt.
/// the current implementation relies on async chat clients and async retrievers.
/// we use generics in order to preserve error types via associated types.
///
/// * `T` - The type of the chat client to be used
/// * `U` - The type of the retriever to be used
#[derive(Debug, TypedBuilder, Clone, PartialEq, Eq)]
pub struct BasicRAGChain<T, U>
where
    T: AsyncChatClient,
    U: AsyncRetriever,
{
    #[builder(default, setter(strip_option))]
    system_prompt: Option<PromptMessage>,
    chat_client: T,
    retriever: U,
}

impl<T, U> BasicRAGChain<T, U>
where
    T: AsyncChatClient,
    U: AsyncRetriever,
{
}

impl<T, U> BasicRAGChain<T, U>
where
    T: AsyncChatClient,
    U: AsyncRetriever,
{
    /// # invoke_chain
    ///
    /// function to execute the RAG chain given a user prompt and a top_k value.
    /// we take the supplied user prompt and retrieve supporting chunks from the retriever.
    /// those chunks are then used to build a new prompt which is then sent to the chat client.
    /// the new prompt then becomes:
    ///
    /// user prompt
    ///
    /// Here is some supporting information:
    ///
    /// chunk1
    ///
    /// chunk2
    ///
    /// ...
    ///
    /// # Arguments
    /// * `user_message` - the user prompt, this will be used to retrieve supporting chunks
    /// * `top_k` - the number of supporting chunks to retrieve
    ///
    /// # Errors
    /// * [`BasicRagChainError`] - if the chat client or retriever fails.
    ///
    /// # Returns
    /// [`PromptMessage`] - the response from the chat client
    pub async fn invoke_chain(
        &self,
        user_message: PromptMessage,
        top_k: NonZeroU32,
    ) -> Result<PromptMessage, RagChainError<T::ErrorType, U::ErrorType>> {
        let content = user_message.content();
        let chunks: Vec<Chunk> = self
            .retriever
            .retrieve(&content, top_k)
            .await
            .map_err(RagChainError::RetrieverError::<T::ErrorType, U::ErrorType>)?;

        let new_prompt: PromptMessage =
            PromptMessage::HumanMessage(build_prompt(user_message, chunks));

        let prompts = match self.system_prompt.clone() {
            None => vec![new_prompt],
            Some(prompt) => vec![prompt, new_prompt],
        };

        let result = self
            .chat_client
            .invoke(prompts)
            .await
            .map_err(RagChainError::ChatClientError::<T::ErrorType, U::ErrorType>)?;

        Ok(result)
    }
}

#[cfg(test)]
mod basic_rag_chain_tests {
    use super::*;
    use async_trait::async_trait;
    use mockall::predicate::eq;
    use mockall::*;
    use std::vec;

    #[tokio::test]
    async fn test_chain_succeeds() {
        const SYSTEM_MESSAGE: &str = "you are a study buddy";
        const USER_MESSAGE: &str = "please tell me about my lecture on operating systems";
        const RAG_CHUNK_1: &str = "data point 1";
        const RAG_CHUNK_2: &str = "data point 2";
        let expected_user_message: String = format!(
            "{}\n{}\n{}\n{}\n",
            USER_MESSAGE, "Here is some supporting information:", RAG_CHUNK_1, RAG_CHUNK_2
        );

        let system_prompt = PromptMessage::SystemMessage(SYSTEM_MESSAGE.into());
        let mut chat_client = MockAsyncChatClient::new();
        let mut retriever = MockAsyncRetriever::new();

        retriever
            .expect_retrieve()
            .with(eq(USER_MESSAGE), eq(NonZeroU32::new(2).unwrap()))
            .returning(|_, _| Ok(vec![Chunk::from(RAG_CHUNK_1), Chunk::from(RAG_CHUNK_2)]));

        chat_client
            .expect_invoke()
            .with(eq(vec![
                system_prompt.clone(),
                PromptMessage::HumanMessage(expected_user_message.into()),
            ]))
            .returning(|_| Ok(PromptMessage::AIMessage("mocked response".into())));

        let chain: BasicRAGChain<MockAsyncChatClient, MockAsyncRetriever> =
            BasicRAGChain::builder()
                .system_prompt(system_prompt)
                .chat_client(chat_client)
                .retriever(retriever)
                .build();

        let user_message = PromptMessage::HumanMessage(USER_MESSAGE.into());

        let result = chain
            .invoke_chain(user_message, NonZeroU32::new(2).unwrap())
            .await
            .unwrap();

        assert_eq!(PromptMessage::AIMessage("mocked response".into()), result)
    }

    mock! {
        AsyncRetriever {}
        #[async_trait]
        impl AsyncRetriever for AsyncRetriever {
            type ErrorType = std::io::Error;
            async fn retrieve(&self, text: &str, top_k: NonZeroU32) -> Result<Vec<Chunk>, <Self as AsyncRetriever>::ErrorType>;
        }
    }

    mock! {
        AsyncChatClient {}
        #[async_trait]
        impl AsyncChatClient for AsyncChatClient {
            type ErrorType = std::io::Error;
            async fn invoke(
                &self,
                prompt_messages: Vec<PromptMessage>,
            ) -> Result<PromptMessage, <Self as AsyncChatClient>::ErrorType>;
        }
    }
}
