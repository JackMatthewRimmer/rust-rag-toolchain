use thiserror::Error;

/// # [`RagChainError`]
///
/// This enum represents the possible errors that can occur when using the BasicRAGChain.
/// It is parametrized over the error types of the chat client and the retriever. this way
/// concrete error types are preserved and can be handled accordingly.
///
/// * `T` - The error type of the chat client
/// * `U` - The error type of the retriever
#[derive(Error, Debug, PartialEq)]
pub enum RagChainError<T, U>
where
    T: std::error::Error + std::fmt::Display,
    U: std::error::Error + std::fmt::Display,
{
    #[error("Chat Client Error: {0}")]
    ChatClientError(T),
    #[error("Retriever Error: {0}")]
    RetrieverError(U),
}

/// # [`ChainError`]
///
/// This enum represents the possible errors that can occur when using the ChatHistoryChain.
/// It is parametrized over the error type of the chat client. this way concrete error types are
/// preserved and can be handled accordingly.
#[derive(Error, Debug, PartialEq)]
pub enum ChainError<T>
where
    T: std::error::Error + std::fmt::Display,
{
    #[error("Chat Client Error: {0}")]
    ChatClientError(T),
}
