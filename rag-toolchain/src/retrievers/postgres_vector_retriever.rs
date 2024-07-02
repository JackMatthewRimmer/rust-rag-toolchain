use crate::clients::AsyncEmbeddingClient;
use crate::common::{Chunk, Chunks, Embedding};
use crate::retrievers::traits::AsyncRetriever;
use pgvector::Vector;
use sqlx::{Pool, Postgres};
use std::error::Error;
use std::num::NonZeroU32;
use thiserror::Error;

/// # [`PostgresVectorRetriever`]
///
/// This struct is a allows for the retrieval of similar text from a postgres database.
/// It is parameterized over a type T which implements the AsyncEmbeddingClient trait.
/// This is because text needs to be embeded before it can be compared to other text.
/// You must connect first create a PostgresVectorStore as this handles connecting to the database.
/// then you can calle .as_retriever() to convert it to retriever.
///
/// # Examples
/// ```
/// use rag_toolchain::retrievers::*;
/// use rag_toolchain::clients::*;
/// use rag_toolchain::common::*;
/// use rag_toolchain::stores::*;
/// use std::num::NonZeroU32;
///
/// async fn retrieve() {
///     let chunk: Chunk = Chunk::new("This is the text you want to retrieve something similar to");
///     let top_k: NonZeroU32 = NonZeroU32::new(5).unwrap();
///     let distance_function: DistanceFunction = DistanceFunction::Cosine;
///     let embedding_model: OpenAIEmbeddingModel = OpenAIEmbeddingModel::TextEmbedding3Small;
///     let client: OpenAIEmbeddingClient = OpenAIEmbeddingClient::try_new(embedding_model).unwrap();
///     let store: PostgresVectorStore = PostgresVectorStore::try_new("table_name", embedding_model).await.unwrap();
///     let retriever: PostgresVectorRetriever<OpenAIEmbeddingClient> = store.as_retriever(client, distance_function);
///     // This will return the top 5 most similar chunks to the input text.
///     let similar_text: Chunks = retriever.retrieve(chunk.content(), top_k).await.unwrap();
/// }
/// ```
pub struct PostgresVectorRetriever<T>
where
    T: AsyncEmbeddingClient,
{
    pool: Pool<Postgres>,
    table_name: String,
    embedding_client: T,
    distance_function: DistanceFunction,
}

impl<T: AsyncEmbeddingClient> PostgresVectorRetriever<T> {
    /// # [`PostgresVectorRetriever::new`]
    /// This constructor is only used internally to allow .as_retriever methods to create a retriever.
    ///
    /// # Arguments
    /// * `pool`: [`sqlx::Pool<Postgres>`] - Which we can use to interact with the database.
    /// * `table_name`: [`String`] - The name of the table which contains the vectors.
    /// * `embedding_client`: [`T`] - An instance of a type which implements the AsyncEmbeddingClient trait.
    ///
    /// # Returns
    /// * [`PostgresVectorRetriever`] the created struct
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

    /// # [`PostgresVectorRetriever::select_row_sql`]
    ///
    /// Helper function to genrate the sql query for a similarity search.
    ///
    /// # Arguments
    /// * `table_name`: &[`str`] - The name of the table to search.
    /// * `distance_function`: [`DistanceFunction`] - The distance function to use.
    ///
    /// # Returns
    /// * [`String`] - The sql query.
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
    /// Implementation of the retrieve function for [`PostgresVectorRetriever`].
    /// This allows us to retrieve similar text from the vector database.
    ///
    /// # Arguments
    /// * `text`: &[`str`] - The text we are searching for similar text against.
    /// * `top_k`: [`NonZeroU32`] - The number of results to return.
    ///
    /// # Errors
    /// * [`PostgresRetrieverError::EmbeddingClientError`] - If the embedding client returns an error.
    /// * [`PostgresRetrieverError::QueryError`] - If there is an error querying the database.
    ///
    /// # Returns
    /// * [`Chunks`] which are the most similar to the input text.
    async fn retrieve(&self, text: &str, top_k: NonZeroU32) -> Result<Chunks, Self::ErrorType> {
        let k: i32 = top_k.get() as i32;
        let chunk: Chunk = Chunk::new(text);
        let embedding: Embedding = self
            .embedding_client
            .generate_embedding(chunk)
            .await
            .map_err(PostgresRetrieverError::EmbeddingClientError)?;

        let query: String = Self::select_row_sql(&self.table_name, self.distance_function.clone());
        let vector: Vec<f32> = embedding.vector();

        let similar_text: Vec<PostgresRow> = sqlx::query_as::<_, PostgresRow>(&query)
            .bind(vector)
            .bind(k)
            .fetch_all(&self.pool)
            .await
            .map_err(PostgresRetrieverError::QueryError)?;

        Ok(similar_text
            .into_iter()
            .map(|row| Chunk::new_with_metadata(row.content, row.metadata))
            .collect())
    }
}

/// # [`DistanceFunction`]
/// This is an enum for the types of distance functions
/// that can be used to compare vectors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DistanceFunction {
    L2,
    Cosine,
    InnerProduct,
}

/// # [`PostgresRow`]
/// Type that represents a row in our defined structure
/// which allows us to use [`sqlx::query_as`].
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

/// # [`PostgresRetrieverError`]
///
/// This error is generic as it is parameterized over the error type of the embedding client.
/// This allows us to avoid dynamic dispatched error types.
#[derive(Error, Debug)]
pub enum PostgresRetrieverError<T: Error> {
    /// If an error occured while trying to embed the text supplied
    /// as an arguement
    #[error("Embedding Client Error: {0}")]
    EmbeddingClientError(T),
    /// If an error occured while doing the similarity search
    #[error("Embedding Retrieving Similar Text: {0}")]
    QueryError(sqlx::Error),
}
