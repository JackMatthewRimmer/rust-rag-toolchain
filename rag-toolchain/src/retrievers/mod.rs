/// # Retrievers
/// This module contains the retrievers for the different vector databases.
/// Once you have connected to a store you can call as_retriever to get a retriever
/// Which allows you given some input text to search for similar text in the store.
mod traits;

#[cfg(feature = "pg_vector")]
mod postgres_vector_retriever;
#[cfg(feature = "pg_vector")]
pub use postgres_vector_retriever::{
    DistanceFunction, PostgresRetrieverError, PostgresVectorRetriever,
};

pub use traits::AsyncRetriever;

// export the trait mocks for use in testing
#[cfg(test)]
pub use traits::MockAsyncRetriever;
