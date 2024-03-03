use crate::clients::AsyncEmbeddingClient;
use crate::common::{Chunk, Embedding, EmbeddingModel};
use crate::retrievers::{DistanceFunction, PostgresVectorRetriever};
use crate::stores::traits::EmbeddingStore;
use sqlx::postgres::{PgPoolOptions, PgQueryResult};
use sqlx::Error as SqlxError;
use sqlx::{postgres::PgArguments, Pool, Postgres};
use std::env::{self, VarError};
use std::error::Error;
use std::fmt::{Display, Formatter};

use dotenv::dotenv;

/// # [`PostgresVectorStore`]
///
/// This struct is used to store and as a retriever information needed to connect to a Postgres database
/// and should be passed to an embedding task as a destination for the data to be stored. If a user
/// wants to do additional operations such as setting an index on the create table they can use the connection
/// pool which is accesible and work with that directly.
///
/// If a table already exists with the same name, the table will be dropped and recreated.
///
/// # Required Environment Variables
///
/// * POSTGRES_USERNAME: The username to connect to the database with
/// * POSTGRES_PASSWORD: The password to connect to the database with
/// * POSTGRES_HOST: The host to connect to the database with
/// * POSTGRES_DATABASE: The database to connect with
///
/// Place these variables in a .env file in the root of your project.
///
/// # Output table format
/// Columns: | id (int) | content (text) | embedding (vector) |
#[derive(Debug, Clone)]
pub struct PostgresVectorStore {
    /// We make the pool public incase users want to
    /// do extra operations on the database
    pool: Pool<Postgres>,
    table_name: String,
}

impl PostgresVectorStore {
    /// # [`PostgresVectorStore::try_new`]
    ///
    /// # Arguments
    /// * `db_name` - The name of the table to store the embeddings in.
    ///
    /// # Errors
    /// * [`PostgresVectorError::EnvVarError`] if the required environment variables are not set
    /// * [`PostgresVectorError::ConnectionError`] if the connection to the database could not be established
    /// * [`PostgresVectorError::TableCreationError`] if the table could not be created
    ///
    /// # Returns
    /// [`PostgresVectorStore`] if the connection and table creation is successful
    pub async fn try_new(
        table_name: &str,
        embedding_model: impl EmbeddingModel,
    ) -> Result<Self, PostgresVectorError> {
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
            .map_err(PostgresVectorError::ConnectionError)?;

        // Create the table
        PostgresVectorStore::create_table(&pool, table_name, embedding_diminsions)
            .await
            .map_err(PostgresVectorError::TableCreationError)?;

        Ok(PostgresVectorStore {
            pool,
            table_name: table_name.into(),
        })
    }

    /// # [`PostgresVectorStore::try_new_with_pool`]
    ///
    /// This function allows us to create a new PostgresVectorStore with a pre-existing connection pool
    /// this is more useful in cases where we want to have multiple stores for different tables all sharing
    /// the same connection pool.
    ///
    /// # Arguments
    /// * `pool` - The connection pool to use to connect to the database
    /// * `table_name` - The name of the table to store the embeddings in
    /// * `embedding_model` - The embedding model to use to store the embeddings
    ///
    /// # Errors
    /// * [`PostgresVectorError::TableCreationError`] if the table could not be created
    ///
    /// # Returns
    /// [`PostgresVectorStore`] if the table creation is successful
    pub async fn try_new_with_pool(
        pool: Pool<Postgres>,
        table_name: &str,
        embedding_model: impl EmbeddingModel,
    ) -> Result<Self, PostgresVectorError> {
        let embedding_diminsions = embedding_model.metadata().dimensions;

        // Create the table
        PostgresVectorStore::create_table(&pool, table_name, embedding_diminsions)
            .await
            .map_err(PostgresVectorError::TableCreationError)?;

        Ok(PostgresVectorStore {
            pool,
            table_name: table_name.into(),
        })
    }

    /// # [`PostgresVectorStore::get_pool`]
    ///
    /// Getter for the internal connection pool
    ///
    /// # Returns
    /// [`Pool`] - The connection pool
    pub fn get_pool(&self) -> Pool<Postgres> {
        self.pool.clone()
    }

    /// # [`PostgresVectorStore::as_retriever`]
    ///
    /// This function allows us to convert the store into a retriever.
    /// Note that the returned retriever is bound to the same table as the store.
    ///
    /// # Arguments
    /// * `embedding_client` - The client we use to embed income text before the
    ///                        similarity search
    /// * `distance_function` - The distance function to use to compare the embeddings
    ///
    /// # Returns
    /// [`PostgresVectorRetriever`] - The retriever that can be used to search for similar text
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
    /// Allows us to check the connection to a database and store the connection pool
    ///
    /// # Arguments
    /// * `connection_string` - The connection string to use to connect to the database
    /// * `rt` - The runtime to use to connect to the database
    ///
    /// # Errors
    /// * [`Error`] if the connection could not be established
    ///
    /// # Returns
    /// * [`Pool`] which can be used to query the database
    /// * [`SqlxError`] if the connection could not be established
    async fn connect(connection_string: &str) -> Result<Pool<Postgres>, SqlxError> {
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
    /// * `table_name` - The name of the table to create
    /// * `pool` - The connection pool to use to create the table
    /// * `rt` - The runtime to use to create the table
    ///
    /// # Errors
    /// * [`Error`] if the table could not be created
    ///
    /// # Returns
    /// * [`PgQueryResult`] which can be used to check if the table was created successfully
    /// * [`SqlxError`] if the table could not be created
    async fn create_table(
        pool: &Pool<Postgres>,
        table_name: &str,
        vector_dimension: usize,
    ) -> Result<PgQueryResult, SqlxError> {
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

    fn insert_row_sql(table_name: &str) -> String {
        format!(
            "INSERT INTO {} (content, embedding, metadata) VALUES ($1, $2, $3)",
            table_name
        )
    }

    fn bind_to_query(
        query: &str,
        embedding: (Chunk, Embedding),
    ) -> sqlx::query::Query<'_, Postgres, PgArguments> {
        let (content, embedding) = embedding;
        let text: String = content.clone().into();
        let vector: Vec<f32> = embedding.clone().into();
        let metadata = content.metadata();
        sqlx::query(query).bind(text).bind(vector).bind(metadata)
    }
}

impl EmbeddingStore for PostgresVectorStore {
    type ErrorType = PostgresVectorError;
    /// # [`PostgresVectorStore::store`]
    ///
    /// # Arguments
    /// * `embeddings` - A tuple containing the content and the embedding to store
    ///
    /// # Errors
    /// * [`PostgresVectorError::InsertError`] if the insert fails
    ///
    /// # Returns
    /// * [`()`] if the insert succeeds
    async fn store(&self, embedding: (Chunk, Embedding)) -> Result<(), PostgresVectorError> {
        let query: String = PostgresVectorStore::insert_row_sql(&self.table_name);
        Self::bind_to_query(&query, embedding)
            .execute(&self.pool)
            .await
            .map_err(PostgresVectorError::InsertError)?;
        Ok(())
    }

    /// # [`PostgresVectorStore::store_batch`]
    ///
    /// # Arguments
    /// * `embeddings` - A vector of tuples containing the content and the embedding to store
    ///
    /// # Errors
    /// * [`PostgresVectorError::TransactionError`] if the transaction fails
    ///
    /// # Returns
    /// * [`()`] if the transaction succeeds
    async fn store_batch(
        &self,
        embeddings: Vec<(Chunk, Embedding)>,
    ) -> Result<(), PostgresVectorError> {
        let query: String = PostgresVectorStore::insert_row_sql(&self.table_name);
        let mut transaction = self
            .pool
            .begin()
            .await
            .map_err(PostgresVectorError::TransactionError)?;

        for embedding in embeddings {
            Self::bind_to_query(&query, embedding)
                .execute(&mut *transaction)
                .await
                .map_err(PostgresVectorError::InsertError)?;
        }

        transaction
            .commit()
            .await
            .map_err(PostgresVectorError::TransactionError)?;
        Ok(())
    }
}

/// # [`PostgresVectorError`]
/// This Error enum wraps all the errors that can occur when using
/// the PgVector struct with contextual meaning
#[derive(Debug)]
pub enum PostgresVectorError {
    /// Error when an environment variable is not set
    EnvVarError(VarError),
    /// Error when the connection to the database could not be established
    ConnectionError(SqlxError),
    /// Error when the table could not be created
    TableCreationError(SqlxError),
    /// Error when calling [`PostgresVectorStore::store()`] fails
    InsertError(SqlxError),
    /// Error when calling [`PostgresVectorStore::store_batch()`] fails
    TransactionError(SqlxError),
}
impl Error for PostgresVectorError {}
impl From<VarError> for PostgresVectorError {
    fn from(error: VarError) -> Self {
        PostgresVectorError::EnvVarError(error)
    }
}

impl Display for PostgresVectorError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            PostgresVectorError::EnvVarError(error) => {
                write!(f, "Environment variable error: {}", error)
            }
            PostgresVectorError::ConnectionError(error) => {
                write!(f, "Connection error: {}", error)
            }
            PostgresVectorError::TableCreationError(error) => {
                write!(f, "Table creation error: {}", error)
            }
            PostgresVectorError::InsertError(error) => {
                write!(f, "Upsert error: {}", error)
            }
            PostgresVectorError::TransactionError(error) => {
                write!(f, "Transaction error: {}", error)
            }
        }
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
        assert!(matches!(result, PostgresVectorError::EnvVarError(_)));

        std::env::set_var("POSTGRES_USER", "postgres");
        let result = PostgresVectorStore::try_new("test", TextEmbeddingAda002)
            .await
            .unwrap_err();
        assert!(matches!(result, PostgresVectorError::EnvVarError(_)));

        std::env::set_var("POSTGRES_PASSWORD", "postgres");
        let result = PostgresVectorStore::try_new("test", TextEmbeddingAda002)
            .await
            .unwrap_err();
        assert!(matches!(result, PostgresVectorError::EnvVarError(_)));

        std::env::set_var("POSTGRES_HOST", "localhost");
        let result = PostgresVectorStore::try_new("test", TextEmbeddingAda002)
            .await
            .unwrap_err();
        assert!(matches!(result, PostgresVectorError::EnvVarError(_)));

        std::env::set_var("POSTGRES_DATABASE", "postgres");
        let result = PostgresVectorStore::try_new("test", TextEmbeddingAda002)
            .await
            .unwrap_err();
        assert!(matches!(result, PostgresVectorError::ConnectionError(_)));
    }
}
