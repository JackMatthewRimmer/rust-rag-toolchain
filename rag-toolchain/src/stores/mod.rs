#[cfg(feature = "pg_vector")]
mod postgres_vector_store;
mod traits;

#[cfg(feature = "pg_vector")]
pub use postgres_vector_store::{
    DistanceFunction, IndexType, PostgresVectorError, PostgresVectorStore,
};
pub use traits::EmbeddingStore;
