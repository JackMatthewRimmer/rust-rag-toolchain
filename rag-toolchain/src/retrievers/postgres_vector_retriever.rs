use crate::clients::traits::AsyncEmbeddingClient;
use crate::common::types::Chunk;
use crate::retrievers::traits::AsyncRetriever;
use async_trait::async_trait;
use sqlx::{Pool, Postgres};

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
            embedding_client: embedding_client,
        }
    }
}

#[async_trait]
impl<T> AsyncRetriever for PostgresVectorRetriever<T>
where
    T: AsyncEmbeddingClient,
{
    type ErrorType = T::ErrorType;
    async fn retrieve(&self, text: &str) -> Result<Chunk, Self::ErrorType> {
        let (content, embedding) = self
            .embedding_client
            .generate_embedding(text.into())
            .await?;
        let embedding_query: String = format!(
            "
            SELECT text FROM {} ORDER BY embedding <=> $1::vector LIMIT 1",
            &self.table_name
        );
        let similar_text: String = sqlx::query_as(&embedding_query)
            .bind(embedding.into())
            .fetch_one(&self.pool)
            .await?;
        Ok(Chunk::from(similar_text))
    }
}
