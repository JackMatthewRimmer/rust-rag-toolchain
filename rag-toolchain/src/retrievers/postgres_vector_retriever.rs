use crate::clients::traits::AsyncEmbeddingClient;
use crate::common::types::Chunk;
use crate::retrievers::traits::AsyncRetriever;
use pgvector::Vector;
use async_trait::async_trait;
use sqlx::postgres::PgRow;
use sqlx::Error as SqlxError;
use sqlx::{Pool, Postgres, Row};
use std::error::Error;
use std::fmt::{Display, Formatter};

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
}

#[async_trait]
impl<T> AsyncRetriever for PostgresVectorRetriever<T>
where
    T: AsyncEmbeddingClient + Sync,
    T::ErrorType: 'static,
{
    type ErrorType = PostgresRetrieverError<T::ErrorType>;
    async fn retrieve(&self, text: &str) -> Result<Chunk, Self::ErrorType> {
        let (_, embedding) = self
            .embedding_client
            .generate_embedding(text.into())
            .await
            .map_err(PostgresRetrieverError::EmbeddingClientError)?;
        let mapped_embedding: Vector = Vector::from(embedding.embedding().to_vec());
        let embedding_query: String = format!(
            "
            SELECT content FROM {} ORDER BY embedding <-> $1 LIMIT 1",
            &self.table_name
        );
        let similar_text: Vec<PgRow> = sqlx::query(&embedding_query)
            .bind(mapped_embedding)
            .fetch_all(&self.pool)
            .await
            .map_err(PostgresRetrieverError::QueryError)?;

        println!("Similar text: {:?}", similar_text.iter().map(|x| x.get::<String, _>("content")).collect::<Vec<String>>());

        Ok(Chunk::from(similar_text[0].get::<String, _>("content")))
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
