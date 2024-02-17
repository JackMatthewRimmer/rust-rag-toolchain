#[cfg(feature = "pg_vector")]
mod postgres_vector_store;
mod traits;

#[cfg(feature = "pg_vector")]
pub use postgres_vector_store::{
    DistanceFunction, NoIndex, PostgresVectorError, PostgresVectorStore, HNSW, IVFFLAT,
};
pub use traits::EmbeddingStore;
