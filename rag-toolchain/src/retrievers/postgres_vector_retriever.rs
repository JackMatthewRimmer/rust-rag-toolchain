use crate::clients::traits::AsyncEmbeddingClient;
use crate::common::types::{Chunk, Embedding};
use crate::retrievers::traits::AsyncRetriever;
use async_trait::async_trait;
use sqlx::postgres::PgRow;
use sqlx::Error as SqlxError;
use sqlx::{Pool, Postgres, Row};
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::num::NonZeroU32;

/// # PostgresVectorRetriever
/// 
/// This struct is a allows for the retrieval of similar text from a postgres database.
/// It is parameterized over a type T which implements the AsyncEmbeddingClient trait.
/// This is because text needs to be embeded before it can be compared to other text.
pub struct PostgresVectorRetriever<T>
where
    T: AsyncEmbeddingClient,
{
    pub pool: Pool<Postgres>,
    table_name: String,
    embedding_client: T,
}

/// # PostgresVectorRetriever
/// 
/// This struct is a allows for the retrieval of similar text from a postgres database.
impl<T: AsyncEmbeddingClient> PostgresVectorRetriever<T> {
    /// # new
    /// This new function should be called the a vectors stores as_retriver() function.
    /// 
    /// # Arguments 
    /// * `pool` - A sqlx::Pool<Postgres> which is used to connect to the database.
    /// * `table_name` - The name of the table which contains the vectors.
    /// * `embedding_client` - An instance of a type which implements the AsyncEmbeddingClient trait.
    /// 
    /// # Returns
    /// * A PostgresVectorRetriever
    pub(crate) fn new(pool: Pool<Postgres>, table_name: String, embedding_client: T) -> Self {
        PostgresVectorRetriever {
            pool,
            table_name,
            embedding_client,
        }
    }
}

/// # AsyncRetriever
/// 
/// This trait is implemented for PostgresVectorRetriever.
#[async_trait]
impl<T> AsyncRetriever for PostgresVectorRetriever<T>
where
    T: AsyncEmbeddingClient + Sync,
    T::ErrorType: 'static,
{
    // We parameterize over the error type of the embedding client.
    type ErrorType = PostgresRetrieverError<T::ErrorType>;

    /// # retrieve
    /// 
    /// Implementation of the retrieve function for PostgresVectorRetriever.
    /// 
    /// # Arguments
    /// * `text` - The text to find similar text for.
    /// * `number_of_results` - The number of results to return.
    /// 
    /// # Errors
    /// * [`PostgresRetrieverError::EmbeddingClientError`] - If the embedding client returns an error.
    /// * [`PostgresRetrieverError::QueryError`] - If there is an error querying the database.
    /// 
    /// # Returns
    /// * A [`Vec<Chunk>`] which are the most similar to the input text.
    async fn retrieve(
        &self,
        text: &str,
        number_of_results: NonZeroU32,
    ) -> Result<Vec<Chunk>, Self::ErrorType> {
        let n: u32 = number_of_results.get();
        let (_, embedding): (_, Embedding) = self
            .embedding_client
            .generate_embedding(text.into())
            .await
            .map_err(PostgresRetrieverError::EmbeddingClientError)?;

        let query: String = format!(
            "
            SELECT content FROM {} ORDER BY embedding <=> $1::vector LIMIT $2",
            &self.table_name
        );

        let similar_text: Vec<PgRow> = sqlx::query(&query)
            .bind(embedding.embedding().to_vec())
            .bind(n as i32)
            .fetch_all(&self.pool)
            .await
            .map_err(PostgresRetrieverError::QueryError)?;

        let n_rows: Vec<Chunk> = similar_text
            .iter()
            .take(n as usize)
            .map(|row| Chunk::from(row.get::<String, _>("content")))
            .collect();
        Ok(n_rows)
    }
}

/// # PostgresRetrieverError
/// 
/// This error is generic as it is parameterized over the error type of the embedding client.
/// This allows us to avoid dynamic dispatched error types.
#[derive(Debug)]
pub enum PostgresRetrieverError<T: Error> {
    EmbeddingClientError(T),
    QueryError(SqlxError),
}
impl<T: Error> Error for PostgresRetrieverError<T> {}
impl<T: Error> Display for PostgresRetrieverError<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            PostgresRetrieverError::EmbeddingClientError(error) => {
                write!(f, "Embedding Client Error: {}", *error)
            }
            PostgresRetrieverError::QueryError(error) => {
                write!(f, "Error retrieving similar text: {}", error)
            }
        }
    }
}
