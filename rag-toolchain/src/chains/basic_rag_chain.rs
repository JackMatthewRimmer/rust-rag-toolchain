use crate::{
    clients::{AsyncChatClient, PromptMessage},
    common::Chunk,
    retrievers::AsyncRetriever,
};
use std::{
    error::Error,
    fmt::{Display, Formatter},
    num::NonZeroU32,
};

pub struct BasicRAGChain<T, U>
where
    T: AsyncChatClient,
    U: AsyncRetriever,
{
    system_prompt: Option<PromptMessage>,
    chat_client: T,
    retriever: U,
}

impl<T, U> BasicRAGChain<T, U>
where
    T: AsyncChatClient,
    U: AsyncRetriever,
{
    fn build_prompt(base_message: PromptMessage, chunks: Vec<Chunk>) -> String {
        let mut builder: String = String::new();
        builder.push_str(&base_message.content());
        builder.push_str("\nHere is some supporting information:\n");
        for chunk in chunks {
            builder.push_str(&format!("{}\n", chunk.chunk()))
        }
        builder
    }
}

impl<T, U> BasicRAGChain<T, U>
where
    T: AsyncChatClient,
    U: AsyncRetriever,
{
    async fn invoke_chain(
        &self,
        user_message: PromptMessage,
        top_k: NonZeroU32,
    ) -> Result<PromptMessage, BasicRagChainError<T::ErrorType, U::ErrorType>> {
        let content = user_message.content();
        let chunks: Vec<Chunk> = self
            .retriever
            .retrieve(&content, top_k)
            .await
            .map_err(BasicRagChainError::RetrieverError::<T::ErrorType, U::ErrorType>)?;

        let new_prompt: PromptMessage =
            PromptMessage::HumanMessage(Self::build_prompt(user_message, chunks));

        let prompts = match self.system_prompt.clone() {
            None => vec![new_prompt],
            Some(prompt) => vec![prompt, new_prompt],
        };

        let result = self
            .chat_client
            .invoke(prompts)
            .await
            .map_err(BasicRagChainError::ChatClientError::<T::ErrorType, U::ErrorType>)?;

        Ok(result)
    }
}

#[derive(Debug, PartialEq)]
pub enum BasicRagChainError<T, U>
where
    T: Error + Display,
    U: Error + Display,
{
    ChatClientError(T),
    RetrieverError(U),
}

impl<T, U> Error for BasicRagChainError<T, U>
where
    T: Error + Display,
    U: Error + Display,
{
}
impl<T, U> Display for BasicRagChainError<T, U>
where
    T: Error + Display,
    U: Error + Display,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ChatClientError(e) => std::fmt::Display::fmt(&e, f),
            Self::RetrieverError(e) => std::fmt::Display::fmt(&e, f),
        }
    }
}
