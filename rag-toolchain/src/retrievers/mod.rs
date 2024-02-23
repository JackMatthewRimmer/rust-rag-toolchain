mod postgres_vector_retriever;
mod traits;

pub use postgres_vector_retriever::{
    DistanceFunction, PostgresRetrieverError, PostgresVectorRetriever,
};
pub use traits::AsyncRetriever;

// export the trait mocks for use in testing
#[cfg(test)]
pub use traits::MockAsyncRetriever;
