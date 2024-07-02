use crate::clients::AsyncEmbeddingClient;
use crate::common::{Chunk, Embedding, EmbeddingModel};
use crate::retrievers::{DistanceFunction, PostgresVectorRetriever};
use crate::stores::traits::EmbeddingStore;
use sqlx::postgres::{PgPoolOptions, PgQueryResult};
use sqlx::{postgres::PgArguments, Pool, Postgres};
use std::env::{self, VarError};
use thiserror::Error;

use dotenv::dotenv;

/// # [`PostgresVectorStore`]
///
/// This is the implementation of [`EmbeddingStore`] for a Postgres database with the
/// pgvector extension enabled. This store takes a table name and an embedding model.
/// If a table already exists with the same name and does not have the expected columns
/// any calls to [`PostgresVectorStore::store`] or [`PostgresVectorStore::store_batch`]
/// will fail.
///
/// # Required Environment Variables
///
/// * POSTGRES_USERNAME: The username to connect to the database with
/// * POSTGRES_PASSWORD: The password to connect to the database with
/// * POSTGRES_HOST: The host to connect to the database with
/// * POSTGRES_DATABASE: The database to connect with
///
/// # Output table format
/// Columns: | id (int) | content (text) | embedding (vector) | metadata (jsonb) |
///
/// # Examples
/// ```
/// use rag_toolchain::stores::*;
/// use rag_toolchain::common::*;
///
/// async fn store(embeddings: Vec<Embedding>) {
///     let embedding_model: OpenAIEmbeddingModel = OpenAIEmbeddingModel::TextEmbedding3Small;
///     let table_name: &str = "table_name";
///     let store: PostgresVectorStore = PostgresVectorStore::try_new(table_name, embedding_model)
///         .await.unwrap();
///     store.store_batch(embeddings).await.unwrap();
/// }
/// ```
#[derive(Debug, Clone)]
pub struct PostgresVectorStore {
    /// We hold a connection pool to the database
    pool: Pool<Postgres>,
    /// The name of the table we are operating on
    table_name: String,
}

impl PostgresVectorStore {
    /// # [`PostgresVectorStore::try_new`]
    ///
    /// This constructor is used to create a new PostgresVectorStore. It will read the required
    /// environment variables in. Try and connect to your postgres database and then create a table
    /// with the given name and the expected columns. If the table already exists with the same name
    /// it will not be re-created.
    ///
    /// # Arguments
    /// * `table_name`: &[`str`] - The name of the table to store the embeddings in.
    /// * `embedding_model`: impl [`EmbeddingModel`] - The embedding model to use to store the embeddings
    ///
    /// # Errors
    /// * [`PostgresVectorError::EnvVarError`] if the required environment variables are not set.
    /// * [`PostgresVectorError::ConnectionError`] if the connection to the database could not be established.
    /// * [`PostgresVectorError::TableCreationError`] if the table could not be created.
    ///
    /// # Returns
    /// * [`PostgresVectorStore`] if the connection and table creation is successful
    pub async fn try_new(
        table_name: &str,
        embedding_model: impl EmbeddingModel,
    ) -> Result<Self, PostgresVectorStoreError> {
        dotenv().ok();
        let username: String = env::var("POSTGRES_USER")?;
        let password: String = env::var("POSTGRES_PASSWORD")?;
        let host: String = env::var("POSTGRES_HOST")?;
        let db_name: String = env::var("POSTGRES_DATABASE")?;

        let embedding_diminsions = embedding_model.metadata().dimensions;
        let connection_string =
            format!("postgres://{}:{}@{}/{}", username, password, host, db_name);

        // Connect to the database
        let pool = PostgresVectorStore::connect(&connection_string)
            .await
            .map_err(PostgresVectorStoreError::ConnectionError)?;

        // Create the table
        PostgresVectorStore::create_table(&pool, table_name, embedding_diminsions)
            .await
            .map_err(PostgresVectorStoreError::TableCreationError)?;

        Ok(PostgresVectorStore {
            pool,
            table_name: table_name.into(),
        })
    }

    /// # [`PostgresVectorStore::try_new_with_pool`]
    ///
    /// This is an alternative constructor that allows you to pass in a connection pool.
    /// This was added as it may be the case people want to establish one connection pool
    /// and then shared it across multiple [`PostgresVectorStore`]s managing different tables.
    ///
    /// # Arguments
    /// * `pool`: [`sqlx::Pool<Postgres>`] - a pre established connection pool.
    /// * `table_name`: &[`str`] - The name of the table to store the embeddings in.
    /// * `embedding_model`: impl[`EmbeddingModel`] - The embedding model used for the genrated embeddings.
    ///
    /// # Errors
    /// * [`PostgresVectorError::TableCreationError`] if the table could not be created
    ///
    /// # Returns
    /// * [`PostgresVectorStore`] if the table creation is successful.
    pub async fn try_new_with_pool(
        pool: Pool<Postgres>,
        table_name: &str,
        embedding_model: impl EmbeddingModel,
    ) -> Result<Self, PostgresVectorStoreError> {
        let embedding_diminsions = embedding_model.metadata().dimensions;

        // Create the table
        PostgresVectorStore::create_table(&pool, table_name, embedding_diminsions)
            .await
            .map_err(PostgresVectorStoreError::TableCreationError)?;

        Ok(PostgresVectorStore {
            pool,
            table_name: table_name.into(),
        })
    }

    /// # [`PostgresVectorStore::get_pool`]
    ///
    /// Getter for the internal connection pool.
    /// This is useful if you want to do any further operations on the database
    /// such as enabling an index on the table.
    ///
    /// # Returns
    /// * [`Pool`] - The connection pool
    pub fn get_pool(&self) -> Pool<Postgres> {
        self.pool.clone()
    }

    /// # [`PostgresVectorStore::as_retriever`]
    ///
    /// This function allows us to convert the store into a retriever.
    /// Note that the returned retriever is bound to the same table as the store.
    ///
    /// # Arguments
    /// * `embedding_client`: [`AsyncEmbeddingClient`] - The client we use to embed
    ///     `                  income text before the similarity search.
    /// * `distance_function`: [`DistanceFunction`] - The distance function to use to
    ///                        compare the embeddings
    ///
    /// # Returns
    /// [`PostgresVectorRetriever`] - The retriever that can be used to search for similar text.
    pub fn as_retriever<T: AsyncEmbeddingClient>(
        &self,
        embedding_client: T,
        distance_function: DistanceFunction,
    ) -> PostgresVectorRetriever<T> {
        PostgresVectorRetriever::new(
            self.pool.clone(),
            self.table_name.clone(),
            embedding_client,
            distance_function,
        )
    }

    /// # [`PostgresVectorStore::connect`]
    /// Allows us to establish a connection to a database and store the connection pool
    ///
    /// # Arguments
    /// * `connection_string`: &[`str`] - The connection string to use to connect to the database
    ///
    /// # Errors
    /// * [`sqlx::Error`] if the connection could not be established.
    ///
    /// # Returns
    /// * [`Pool`] which can be used to query the database
    async fn connect(connection_string: &str) -> Result<Pool<Postgres>, sqlx::Error> {
        let pool: Pool<Postgres> = PgPoolOptions::new()
            .max_connections(5)
            .connect(connection_string)
            .await?;
        Ok(pool)
    }

    /// # [`PostgresVectorStore::create_table`]
    /// We call the create table automatically when the struct is created
    ///
    /// # Arguments
    /// * `pool`: [`sqlx::Pool<Postgres>`] - The connection pool to use to create the table
    /// * `table_name`: &[`str`] - The name of the table to create
    /// * `vector_dimension`: [`usize`] - The dimension of the vector to store
    ///
    /// # Errors
    /// * [`sqlx::Error`] if the table could not be created.
    ///
    /// # Returns
    /// * [`PgQueryResult`] which can be used to check if the table was created successfully
    async fn create_table(
        pool: &Pool<Postgres>,
        table_name: &str,
        vector_dimension: usize,
    ) -> Result<PgQueryResult, sqlx::Error> {
        let statement = format!(
            "CREATE TABLE IF NOT EXISTS {} (
                id SERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                embedding VECTOR({}) NOT NULL,
                metadata JSONB
            )",
            table_name, vector_dimension
        );
        sqlx::query(&statement).execute(pool).await
    }

    /// # [`PostgresVectorStore::insert_row_sql`]
    /// Helper function to generate the sql query for inserting a new row
    ///
    /// # Arguments
    /// * `table_name`: &[`str`] - The name of the table to insert into
    ///
    /// # Returns
    /// * [`String`] - The sql query
    fn insert_row_sql(table_name: &str) -> String {
        format!(
            "INSERT INTO {} (content, embedding, metadata) VALUES ($1, $2, $3)",
            table_name
        )
    }

    /// # [`PostgresVectorStore::bind_to_query`]
    /// Helper function to bind an [`Embedding`] to an [`sqlx::query::Query`]
    /// the retuned query can then have [`sqlx::query::Query::execute`] called on it to
    /// insert the row.
    fn bind_to_query(
        query: &str,
        embedding: Embedding,
    ) -> sqlx::query::Query<'_, Postgres, PgArguments> {
        let chunk: &Chunk = embedding.chunk();
        let text: String = chunk.content().to_string();
        let metadata = chunk.metadata().clone();
        let vector: Vec<f32> = embedding.vector();
        sqlx::query(query).bind(text).bind(vector).bind(metadata)
    }
}

impl EmbeddingStore for PostgresVectorStore {
    type ErrorType = PostgresVectorStoreError;
    /// # [`PostgresVectorStore::store`]
    /// This is done as a single insert statement.
    ///
    /// # Arguments
    /// * `embedding`: [`Embedding`] - to insert
    ///
    /// # Errors
    /// * [`PostgresVectorError::InsertError`] if the insert fails
    ///
    /// # Returns
    /// * [`()`] if the insert succeeds
    async fn store(&self, embedding: Embedding) -> Result<(), PostgresVectorStoreError> {
        let query: String = PostgresVectorStore::insert_row_sql(&self.table_name);
        Self::bind_to_query(&query, embedding)
            .execute(&self.pool)
            .await
            .map_err(PostgresVectorStoreError::InsertError)?;
        Ok(())
    }

    /// # [`PostgresVectorStore::store_batch`]
    /// This is done as a single transaction with multiple insert statements.
    ///
    /// # Arguments
    /// * `embeddings`: [`Vec<Embedding>`] - A vector of embeddings to insert
    ///
    /// # Errors
    /// * [`PostgresVectorError::TransactionError`] if the transaction fails
    ///
    /// # Returns
    /// * [`()`] if the transaction succeeds
    async fn store_batch(
        &self,
        embeddings: Vec<Embedding>,
    ) -> Result<(), PostgresVectorStoreError> {
        let query: String = PostgresVectorStore::insert_row_sql(&self.table_name);
        let mut transaction = self
            .pool
            .begin()
            .await
            .map_err(PostgresVectorStoreError::TransactionError)?;

        for embedding in embeddings {
            Self::bind_to_query(&query, embedding)
                .execute(&mut *transaction)
                .await
                .map_err(PostgresVectorStoreError::InsertError)?;
        }

        transaction
            .commit()
            .await
            .map_err(PostgresVectorStoreError::TransactionError)?;
        Ok(())
    }
}

/// # [`PostgresVectorError`]
/// This Error enum wraps all the errors that can occur when using
/// the PgVector struct with contextual meaning.
#[derive(Error, Debug)]
pub enum PostgresVectorStoreError {
    /// Error when an environment variable is not set
    #[error("Environment Variable Error: {0}")]
    EnvVarError(VarError),
    /// Error when the connection to the database could not be established
    #[error("Connection Error: {0}")]
    ConnectionError(sqlx::Error),
    /// Error when the table could not be created
    #[error("Table Creation Error: {0}")]
    TableCreationError(sqlx::Error),
    /// Error when calling [`PostgresVectorStore::store()`] fails
    #[error("Upsert Error: {0}")]
    InsertError(sqlx::Error),
    /// Error when calling [`PostgresVectorStore::store_batch()`] fails
    #[error("Transaction Error: {0}")]
    TransactionError(sqlx::Error),
}

impl From<VarError> for PostgresVectorStoreError {
    fn from(error: VarError) -> Self {
        PostgresVectorStoreError::EnvVarError(error)
    }
}

#[cfg(all(test, feature = "pg_vector"))]
mod tests {
    use super::*;
    use crate::common::OpenAIEmbeddingModel::TextEmbeddingAda002;

    #[tokio::test]
    async fn test_throws_correct_errors() {
        let result = PostgresVectorStore::try_new("test", TextEmbeddingAda002)
            .await
            .unwrap_err();
        assert!(matches!(result, PostgresVectorStoreError::EnvVarError(_)));

        std::env::set_var("POSTGRES_USER", "postgres");
        let result = PostgresVectorStore::try_new("test", TextEmbeddingAda002)
            .await
            .unwrap_err();
        assert!(matches!(result, PostgresVectorStoreError::EnvVarError(_)));

        std::env::set_var("POSTGRES_PASSWORD", "postgres");
        let result = PostgresVectorStore::try_new("test", TextEmbeddingAda002)
            .await
            .unwrap_err();
        assert!(matches!(result, PostgresVectorStoreError::EnvVarError(_)));

        std::env::set_var("POSTGRES_HOST", "localhost");
        let result = PostgresVectorStore::try_new("test", TextEmbeddingAda002)
            .await
            .unwrap_err();
        assert!(matches!(result, PostgresVectorStoreError::EnvVarError(_)));

        std::env::set_var("POSTGRES_DATABASE", "postgres");
        let result = PostgresVectorStore::try_new("test", TextEmbeddingAda002)
            .await
            .unwrap_err();
        assert!(matches!(
            result,
            PostgresVectorStoreError::ConnectionError(_)
        ));
    }
}
