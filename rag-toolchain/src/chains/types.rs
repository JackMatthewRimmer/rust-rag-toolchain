use std::{
    error::Error,
    fmt::{Display, Formatter},
};

/// # [`RagChainError`] 
///
/// This enum represents the possible errors that can occur when using the BasicRAGChain.
/// It is parametrized over the error types of the chat client and the retriever. this way
/// concrete error types are preserved and can be handled accordingly.
///
/// * `T` - The error type of the chat client
/// * `U` - The error type of the retriever
#[derive(Debug, PartialEq)]
pub enum RagChainError<T, U>
where
    T: Error + Display,
    U: Error + Display,
{
    ChatClientError(T),
    RetrieverError(U),
}

impl<T, U> Error for RagChainError<T, U>
where
    T: Error + Display,
    U: Error + Display,
{
}
impl<T, U> Display for RagChainError<T, U>
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

/// # [`ChainError`] 
///
/// This enum represents the possible errors that can occur when using the ChatHistoryChain.
/// It is parametrized over the error type of the chat client. this way concrete error types are
/// preserved and can be handled accordingly.
#[derive(Debug, PartialEq)]
pub enum ChainError<T>
where
    T: Error + Display,
{
    ChatClientError(T),
}

impl<T> Error for ChainError<T> where T: Error + Display {}
impl<T> Display for ChainError<T>
where
    T: Error + Display,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ChatClientError(e) => std::fmt::Display::fmt(&e, f),
        }
    }
}
