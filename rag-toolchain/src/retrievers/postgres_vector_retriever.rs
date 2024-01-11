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

pub struct PostgresVectorRetriever<T>
where
    T: AsyncEmbeddingClient,
{
    pub pool: Pool<Postgres>,
    table_name: String,
    embedding_client: T,
}

impl<T: AsyncEmbeddingClient> PostgresVectorRetriever<T> {
    pub fn new(pool: Pool<Postgres>, table_name: String, embedding_client: T) -> Self {
        PostgresVectorRetriever {
            pool,
            table_name,
            embedding_client,
        }
    }

    async fn execute_query(
        &self,
        query: String,
        number_of_results: u32,
        embedding: Embedding,
    ) -> Result<Vec<Chunk>, PostgresRetrieverError<T::ErrorType>> {
        let similar_text: Vec<PgRow> = sqlx::query(&query)
            .bind(embedding.embedding().to_vec())
            .bind(number_of_results as i32)
            .fetch_all(&self.pool)
            .await
            .map_err(PostgresRetrieverError::QueryError)?;
        let n_rows: Vec<Chunk> = similar_text
            .iter()
            .take(number_of_results as usize)
            .map(|row| Chunk::from(row.get::<String, _>("content")))
            .collect();
        Ok(n_rows)
    }
}

#[async_trait]
impl<T> AsyncRetriever for PostgresVectorRetriever<T>
where
    T: AsyncEmbeddingClient + Sync,
    T::ErrorType: 'static,
{
    type ErrorType = PostgresRetrieverError<T::ErrorType>;
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
        self.execute_query(query, n, embedding).await
    }

    // TODO: Implement this method properly
    async fn retrieve_with_threshold(
        &self,
        text: &str,
        number_of_results: NonZeroU32,
        threshold: f32,
    ) -> Result<Vec<Chunk>, Self::ErrorType> {
        let n: u32 = number_of_results.get();
        let (_, embedding) = self
            .embedding_client
            .generate_embedding(text.into())
            .await
            .map_err(PostgresRetrieverError::EmbeddingClientError)?;

        // This needs a threshold adding
        let query: String = format!(
            "
            SELECT content FROM {} ORDER BY embedding <=> $1::vector LIMIT $2",
            &self.table_name
        );
        self.execute_query(query, n, embedding).await
    }
}

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
