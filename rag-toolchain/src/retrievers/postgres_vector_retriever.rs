use crate::clients::AsyncEmbeddingClient;
use crate::common::{Chunk, Chunks, Embedding};
use crate::retrievers::traits::AsyncRetriever;
use pgvector::Vector;
use sqlx::Error as SqlxError;
use sqlx::{Pool, Postgres};
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::num::NonZeroU32;

/// # [`PostgresVectorRetriever`]
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
    distance_function: DistanceFunction,
}

impl<T: AsyncEmbeddingClient> PostgresVectorRetriever<T> {
    /// # [`PostgresVectorRetriever::new`]
    /// This new function should be called the a vectors stores as_retriver() function.
    ///
    /// # Arguments
    /// * `pool` - A sqlx::Pool<Postgres> which is used to connect to the database.
    /// * `table_name` - The name of the table which contains the vectors.
    /// * `embedding_client` - An instance of a type which implements the AsyncEmbeddingClient trait.
    ///
    /// # Returns
    /// * A PostgresVectorRetriever
    pub(crate) fn new(
        pool: Pool<Postgres>,
        table_name: String,
        embedding_client: T,
        distance_function: DistanceFunction,
    ) -> Self {
        PostgresVectorRetriever {
            pool,
            table_name,
            embedding_client,
            distance_function,
        }
    }

    fn select_row_sql(table_name: &str, distance_function: DistanceFunction) -> String {
        format!(
            "SELECT id, content, embedding, metadata FROM {} ORDER BY embedding {} $1::vector LIMIT $2",
            table_name,
            distance_function.to_sql_string()
        )
    }
}

impl<T> AsyncRetriever for PostgresVectorRetriever<T>
where
    T: AsyncEmbeddingClient + Sync,
    T::ErrorType: 'static,
{
    // We parameterize over the error type of the embedding client.
    type ErrorType = PostgresRetrieverError<T::ErrorType>;

    /// # [`PostgresVectorRetriever::retrieve`]
    ///
    /// Implementation of the retrieve function for PostgresVectorRetriever.
    /// This is currently doing a cosine similarity search. We intend to support
    /// all the similarity functions supported by postgres in the future.
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
    /// * A [`Chunks`] which are the most similar to the input text.
    async fn retrieve(&self, text: &str, top_k: NonZeroU32) -> Result<Chunks, Self::ErrorType> {
        let k: i32 = top_k.get() as i32;
        let (_, embedding): (_, Embedding) = self
            .embedding_client
            .generate_embedding(text.into())
            .await
            .map_err(PostgresRetrieverError::EmbeddingClientError)?;

        let query: String = Self::select_row_sql(&self.table_name, self.distance_function.clone());

        let similar_text: Vec<PostgresRow> = sqlx::query_as::<_, PostgresRow>(&query)
            .bind(embedding.embedding().to_vec())
            .bind(k)
            .fetch_all(&self.pool)
            .await
            .map_err(PostgresRetrieverError::QueryError)?;

        Ok(similar_text.into_iter().map(|row| Chunk::from(row.content)).collect())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DistanceFunction {
    L2,
    Cosine,
    InnerProduct,
}

/// # [`PostgresRow`]
/// Type that represents a row in our defined structure
#[derive(Debug, Clone, PartialEq, sqlx::FromRow)]
pub struct PostgresRow {
    pub id: i32,
    pub content: String,
    pub embedding: Vector,
    #[sqlx(json)]
    pub metadata: serde_json::Value,
}

impl DistanceFunction {
    pub fn to_sql_string(&self) -> &str {
        match self {
            DistanceFunction::L2 => "<->",
            DistanceFunction::Cosine => "<=>",
            DistanceFunction::InnerProduct => "<#>",
        }
    }
}

impl Display for DistanceFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            DistanceFunction::L2 => write!(f, "L2"),
            DistanceFunction::Cosine => write!(f, "Cosine"),
            DistanceFunction::InnerProduct => write!(f, "InnerProduct"),
        }
    }
}

/// # [`PostgresRetrieverError`]
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
