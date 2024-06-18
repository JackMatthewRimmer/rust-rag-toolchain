use crate::{
    chains::{utils::build_prompt, RagChainError},
    clients::{AsyncChatClient, AsyncStreamedChatClient, PromptMessage},
    common::Chunks,
    retrievers::AsyncRetriever,
};
use std::num::NonZeroU32;
use typed_builder::TypedBuilder;

/// # [`BasicRAGChain`]
///
/// This struct allows for easily executing RAG given a single user prompt.
/// the current implementation relies on async chat clients and async retrievers.
/// we use generics in order to preserve error types via associated types.
///
/// * `T` - The type of the chat client to be used
/// * `U` - The type of the retriever to be used
///
/// # Examples
/// ```
/// use rag_toolchain::clients::*;
/// use rag_toolchain::retrievers::*;
/// use rag_toolchain::chains::*;
/// use rag_toolchain::stores::*;
/// use rag_toolchain::common::*;
/// use std::num::NonZeroU32;
///
/// async fn run_chain() {
///
///    const SYSTEM_MESSAGE: &'static str =
///         "You are to give straight forward answers using the supporting information you are provided";
///
///    let store: PostgresVectorStore =
///    PostgresVectorStore::try_new("embeddings", OpenAIEmbeddingModel::TextEmbeddingAda002)
///        .await
///        .unwrap();
///
///    let embedding_client: OpenAIEmbeddingClient =
///        OpenAIEmbeddingClient::try_new(OpenAIEmbeddingModel::TextEmbeddingAda002).unwrap();
///
///    let retriever: PostgresVectorRetriever<OpenAIEmbeddingClient> =
///        store.as_retriever(embedding_client, DistanceFunction::Cosine);
///
///    let chat_client: OpenAIChatCompletionClient =
///        OpenAIChatCompletionClient::try_new(OpenAIModel::Gpt3Point5Turbo).unwrap();
///
///    let system_prompt: PromptMessage = PromptMessage::SystemMessage(SYSTEM_MESSAGE.into());
///
///    let chain: BasicRAGChain<OpenAIChatCompletionClient, PostgresVectorRetriever<_>> =
///        BasicRAGChain::builder()
///            .system_prompt(system_prompt)
///            .chat_client(chat_client)
///            .retriever(retriever)
///            .build();
///    let user_message: PromptMessage =
///        PromptMessage::HumanMessage("what kind of alcohol does Morwenna drink".into());
///
///    let response = chain
///        .invoke_chain(user_message, NonZeroU32::new(2).unwrap())
///        .await
///        .unwrap();
/// }
/// ```
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
    /// # [`BasicRAGChain::invoke_chain`]
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
    /// * `user_message`: [`PromptMessage`] - the user prompt, this will be used to retrieve supporting chunks
    /// * `top_k`: [`NonZeroU32`] - the number of supporting chunks to retrieve
    ///
    /// # Errors
    /// * [`RagChainError`] - if the chat client or retriever fails.
    ///
    /// # Returns
    /// [`PromptMessage`] - the response from the chat client
    pub async fn invoke_chain(
        &self,
        user_message: PromptMessage,
        top_k: NonZeroU32,
    ) -> Result<PromptMessage, RagChainError<T::ErrorType, U::ErrorType>> {
        let content = user_message.content();
        let chunks: Chunks = self
            .retriever
            .retrieve(content, top_k)
            .await
            .map_err(RagChainError::RetrieverError::<T::ErrorType, U::ErrorType>)?;

        let new_prompt: PromptMessage = build_prompt(&user_message, chunks);

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

/// # [`BasicStreamedRAGChain`]
///
/// This struct allows for easily executing RAG given a single user prompt.
/// the current implementation relies on async streamed chat clients and async retrievers.
/// we use generics in order to preserve error types via associated types.
///
/// * `T` - The type of the streamed chat client to be used
/// * `U` - The type of the retriever to be used
///
/// # Examples
/// ```
/// use rag_toolchain::clients::*;
/// use rag_toolchain::retrievers::*;
/// use rag_toolchain::chains::*;
/// use rag_toolchain::stores::*;
/// use rag_toolchain::common::*;
/// use std::num::NonZeroU32;
///
/// async fn run_chain() {
///
///    const SYSTEM_MESSAGE: &'static str =
///         "You are to give straight forward answers using the supporting information you are provided";
///
///    let store: PostgresVectorStore =
///    PostgresVectorStore::try_new("embeddings", OpenAIEmbeddingModel::TextEmbeddingAda002)
///        .await
///        .unwrap();
///
///    let embedding_client: OpenAIEmbeddingClient =
///        OpenAIEmbeddingClient::try_new(OpenAIEmbeddingModel::TextEmbeddingAda002).unwrap();
///
///    let retriever: PostgresVectorRetriever<OpenAIEmbeddingClient> =
///        store.as_retriever(embedding_client, DistanceFunction::Cosine);
///
///    let chat_client: OpenAIChatCompletionClient =
///        OpenAIChatCompletionClient::try_new(OpenAIModel::Gpt3Point5Turbo).unwrap();
///
///    let system_prompt: PromptMessage = PromptMessage::SystemMessage(SYSTEM_MESSAGE.into());
///
///    let chain: BasicStreamedRAGChain<OpenAIChatCompletionClient, PostgresVectorRetriever<_>> =
///        BasicStreamedRAGChain::builder()
///            .system_prompt(system_prompt)
///            .chat_client(chat_client)
///            .retriever(retriever)
///            .build();
///    let user_message: PromptMessage =
///        PromptMessage::HumanMessage("what kind of alcohol does Morwenna drink".into());
///
///    let stream = chain
///        .invoke_chain(user_message, NonZeroU32::new(2).unwrap())
///        .await
///        .unwrap();
/// }
/// ```
#[derive(Debug, TypedBuilder, Clone, PartialEq, Eq)]
pub struct BasicStreamedRAGChain<T, U>
where
    T: AsyncStreamedChatClient,
    U: AsyncRetriever,
{
    #[builder(default, setter(strip_option))]
    system_prompt: Option<PromptMessage>,
    chat_client: T,
    retriever: U,
}

impl<T, U> BasicStreamedRAGChain<T, U>
where
    T: AsyncStreamedChatClient,
    U: AsyncRetriever,
{
    pub async fn invoke_chain(
        &self,
        user_message: PromptMessage,
        top_k: NonZeroU32,
    ) -> Result<T::Item, RagChainError<T::ErrorType, U::ErrorType>> {
        let content = user_message.content();
        let chunks: Chunks = self
            .retriever
            .retrieve(content, top_k)
            .await
            .map_err(RagChainError::RetrieverError::<T::ErrorType, U::ErrorType>)?;

        let new_prompt: PromptMessage = build_prompt(&user_message, chunks);

        let prompts = match self.system_prompt.clone() {
            None => vec![new_prompt],
            Some(prompt) => vec![prompt, new_prompt],
        };

        let result = self
            .chat_client
            .invoke_stream(prompts)
            .await
            .map_err(RagChainError::ChatClientError::<T::ErrorType, U::ErrorType>)?;

        Ok(result)
    }
}

#[cfg(test)]
mod basic_rag_chain_tests {
    use super::*;
    use crate::{
        clients::{
            ChatCompletionStream, MockAsyncChatClient, MockAsyncStreamedChatClient,
            MockChatCompletionStream,
        },
        common::Chunk,
        retrievers::MockAsyncRetriever,
    };
    use mockall::predicate::eq;
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
            .returning(|_, _| Ok(vec![Chunk::new(RAG_CHUNK_1), Chunk::new(RAG_CHUNK_2)]));

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

    #[tokio::test]
    async fn test_streamed_chain_succeeds() {
        const SYSTEM_MESSAGE: &str = "you are a study buddy";
        const USER_MESSAGE: &str = "please tell me about my lecture on operating systems";
        const RAG_CHUNK_1: &str = "data point 1";
        const RAG_CHUNK_2: &str = "data point 2";
        let expected_user_message: String = format!(
            "{}\n{}\n{}\n{}\n",
            USER_MESSAGE, "Here is some supporting information:", RAG_CHUNK_1, RAG_CHUNK_2
        );

        let system_prompt = PromptMessage::SystemMessage(SYSTEM_MESSAGE.into());
        let mut chat_client = MockAsyncStreamedChatClient::new();
        let mut retriever = MockAsyncRetriever::new();
        retriever
            .expect_retrieve()
            .with(eq(USER_MESSAGE), eq(NonZeroU32::new(2).unwrap()))
            .returning(|_, _| Ok(vec![Chunk::new(RAG_CHUNK_1), Chunk::new(RAG_CHUNK_2)]));

        chat_client
            .expect_invoke_stream()
            .with(eq(vec![
                system_prompt.clone(),
                PromptMessage::HumanMessage(expected_user_message.into()),
            ]))
            .returning(move |_| {
                let mut stream = MockChatCompletionStream::new();
                stream
                    .expect_next()
                    .returning(|| Some(Ok(PromptMessage::AIMessage("mocked response".into()))));
                Ok(stream)
            });

        let chain: BasicStreamedRAGChain<MockAsyncStreamedChatClient, MockAsyncRetriever> =
            BasicStreamedRAGChain::builder()
                .system_prompt(system_prompt)
                .chat_client(chat_client)
                .retriever(retriever)
                .build();

        let user_message = PromptMessage::HumanMessage(USER_MESSAGE.into());

        let mut result = chain
            .invoke_chain(user_message, NonZeroU32::new(2).unwrap())
            .await
            .unwrap();

        assert_eq!(
            result.next().await.unwrap().unwrap(),
            PromptMessage::AIMessage("mocked response".into())
        );
    }
}
