/// # Stores
/// What is a store ?
///
/// A store in the context of this library is a place when you can stores
/// embddings to any of our supported vector stores.
///
/// Once you have stored them you should be able to call as_retriever
/// to get a retriever (See retrievers module) to preform similarity searches
/// on incoming text.
#[cfg(feature = "pg_vector")]
mod postgres_vector_store;
mod traits;

#[cfg(feature = "pg_vector")]
pub use postgres_vector_store::{PostgresVectorStore, PostgresVectorStoreError};
pub use traits::EmbeddingStore;
