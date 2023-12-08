use crate::toolchain_embeddings::embedding_models::HasMetadata;
use crate::toolchain_indexing::traits::EmbeddingStore;
use crate::toolchain_indexing::types::{Chunk, Embedding};
use async_trait::async_trait;
use sqlx::postgres::{PgPoolOptions, PgQueryResult};
use sqlx::Error as SqlxError;
use sqlx::{Pool, Postgres};
use std::env::{self, VarError};
use std::fmt::{Display, Formatter};
use std::vec;

use dotenv::dotenv;

/// # PgVector
///
/// This struct is used to store and retriever information needed to connect to a Postgres database
/// and should be passed to an embedding task as a destination for the data to be stored.
/// Using the will require an async runtime as sqlx is used to connect to the database.
///
/// If a table already exists with the same name, the table will be dropped and recreated.
///
/// # Required Environment Variables
///
/// * POSTGRES_USERNAME: The username to connect to the database with
/// * POSTGRES_PASSWORD: The password to connect to the database with
/// * POSTGRES_HOST: The host to connect to the database with
///
/// Place these variables in a .env file in the root of your project.
///
/// # Output table format
/// Columns: | id (int) | content (text) | embedding (vector) |
#[derive(Debug)]
pub struct PgVectorDB {
    /// We make the pool public incase users want to
    /// do extra operations on the database
    pub pool: Pool<Postgres>,
    table_name: String,
}

impl PgVectorDB {
    /// # new
    /// # Arguments
    /// * `db_name` - The name of the table to store the embeddings in.
    ///
    /// # Errors
    /// * [`PgVectorError::EnvVarError`] if the required environment variables are not set
    /// * [`PgVectorError::ConnectionError`] if the connection to the database could not be established
    /// * [`PgVectorError::TableCreationError`] if the table could not be created
    ///
    /// # Returns
    /// the constructed [`PgVector`] struct
    pub async fn new(
        table_name: &str,
        embedding_model: impl HasMetadata,
    ) -> Result<Self, PgVectorError> {
        dotenv().ok();
        let username: String = env::var("POSTGRES_USER")?;
        let password: String = env::var("POSTGRES_PASSWORD")?;
        let host: String = env::var("POSTGRES_HOST")?;
        let db_name: String = env::var("POSTGRES_DATABASE")?;
        let table_name: &str = table_name;

        let embedding_diminsions = embedding_model.metadata().dimensions;
        let connection_string =
            format!("postgres://{}:{}@{}/{}", username, password, host, db_name);

        // Connect to the database
        let pool = PgVectorDB::connect(&connection_string)
            .await
            .map_err(PgVectorError::ConnectionError)?;

        // Create the table
        PgVectorDB::create_table(pool.clone(), table_name, embedding_diminsions)
            .await
            .map_err(PgVectorError::TableCreationError)?;

        Ok(PgVectorDB {
            pool,
            table_name: table_name.into(),
        })
    }

    /// # connect
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
    /// * [`Pool<Postgres>`] which can be used to query the database
    /// * [`Error`] if the connection could not be established
    async fn connect(connection_string: &str) -> Result<Pool<Postgres>, SqlxError> {
        let pool: Pool<Postgres> = PgPoolOptions::new()
            .max_connections(5)
            .connect(connection_string)
            .await?;
        Ok(pool)
    }

    /// # create_table
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
    /// * [`Error`] if the table could not be created
    async fn create_table(
        pool: Pool<Postgres>,
        table_name: &str,
        vector_dimension: usize,
    ) -> Result<PgQueryResult, SqlxError> {
        let query = format!(
            "
            CREATE TABLE IF NOT EXISTS {} (
                id SERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                embedding vector({}) 
            )",
            table_name, vector_dimension
        );
        sqlx::query(&query).execute(&pool).await
    }
}

#[async_trait]
impl EmbeddingStore for PgVectorDB {
    type ErrorType = PgVectorError;
    /// # store
    ///
    /// # Arguments
    /// * `embeddings` - A tuple containing the content and the embedding to store
    ///
    /// # Errors
    /// * [`PgVectorError::InsertError`] if the insert fails
    ///
    /// # Returns
    /// * [`Ok(())`] if the insert succeeds
    /// * [`PgVectorError::InsertError`] if the insert fails
    async fn store(&self, embeddings: (Chunk, Embedding)) -> Result<(), PgVectorError> {
        let (content, embedding) = embeddings;
        let text: String = content.into();
        let vector: Vec<f32> = embedding.into();

        let query = format!(
            "
            INSERT INTO {} (content, embedding) VALUES ($1, $2::vector)",
            &self.table_name
        );

        sqlx::query(&query)
            .bind(text)
            .bind(vector)
            .execute(&self.pool)
            .await
            .map_err(PgVectorError::InsertError)?;
        Ok(())
    }

    /// # store_batch
    ///
    /// # Arguments
    /// * `embeddings` - A vector of tuples containing the content and the embedding to store
    ///
    /// # Errors
    /// * [`PgVectorError::TransactionError`] if the transaction fails
    ///
    /// # Returns
    /// * [`Ok(())`] if the transaction succeeds
    /// * [`PgVectorError::TransactionError`] if the transaction fails
    async fn store_batch(&self, embeddings: Vec<(Chunk, Embedding)>) -> Result<(), PgVectorError> {
        let query = format!(
            "
            INSERT INTO {} (content, embedding) VALUES ($1, $2::vector)",
            &self.table_name
        );

        let mut transaction = self
            .pool
            .begin()
            .await
            .map_err(PgVectorError::TransactionError)?;

        for (content, embedding) in embeddings {
            let text: String = content.into();
            let vector: Vec<f32> = embedding.into();
            sqlx::query(&query)
                .bind(text)
                .bind(vector)
                .execute(&mut *transaction)
                .await
                .map_err(PgVectorError::InsertError)?;
        }
        transaction
            .commit()
            .await
            .map_err(PgVectorError::TransactionError)?;
        Ok(())
    }
}

/// # PgVectorError
/// This Error enum wraps all the errors that can occur when using
/// the PgVector struct with contextual meaning
#[derive(Debug)]
pub enum PgVectorError {
    /// Error when an environment variable is not set
    EnvVarError(VarError),
    /// Error when the connection to the database could not be established
    ConnectionError(SqlxError),
    /// Error when the table could not be created
    TableCreationError(SqlxError),
    /// Error when calling [`PgVector::store()`] fails
    InsertError(SqlxError),
    /// Error when calling [`PgVector::store_batch()`] fails
    TransactionError(SqlxError),
}
impl std::error::Error for PgVectorError {}
impl From<VarError> for PgVectorError {
    fn from(error: VarError) -> Self {
        PgVectorError::EnvVarError(error)
    }
}

impl Display for PgVectorError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            PgVectorError::EnvVarError(error) => {
                write!(f, "Environment variable error: {}", error)
            }
            PgVectorError::ConnectionError(error) => {
                write!(f, "Connection error: {}", error)
            }
            PgVectorError::TableCreationError(error) => {
                write!(f, "Table creation error: {}", error)
            }
            PgVectorError::InsertError(error) => {
                write!(f, "Upsert error: {}", error)
            }
            PgVectorError::TransactionError(error) => {
                write!(f, "Transaction error: {}", error)
            }
        }
    }
}

#[cfg(all(test, feature = "pg_vector"))]
mod tests {
    use super::*;
    use crate::toolchain_embeddings::embedding_models::OpenAIEmbeddingModel::TextEmbeddingAda002;

    #[tokio::test]
    async fn test_throws_env_var_error_user() {
        let result = PgVectorDB::new("test", TextEmbeddingAda002)
            .await
            .unwrap_err();
        assert!(matches!(result, PgVectorError::EnvVarError(_)));
    }

    #[tokio::test]
    async fn test_throws_env_var_error_password() {
        std::env::set_var("POSTGRES_USER", "postgres");
        let result = PgVectorDB::new("test", TextEmbeddingAda002)
            .await
            .unwrap_err();
        assert!(matches!(result, PgVectorError::EnvVarError(_)));
    }

    #[tokio::test]
    async fn test_throws_env_var_error_host() {
        std::env::set_var("POSTGRES_USER", "postgres");
        std::env::set_var("POSTGRES_PASSWORD", "postgres");
        let result = PgVectorDB::new("test", TextEmbeddingAda002)
            .await
            .unwrap_err();

        assert!(matches!(result, PgVectorError::EnvVarError(_)));
    }

    #[tokio::test]
    async fn test_throws_env_var_error_database() {
        std::env::set_var("POSTGRES_USER", "postgres");
        std::env::set_var("POSTGRES_PASSWORD", "postgres");
        std::env::set_var("POSTGRES_HOST", "localhost");
        let result = PgVectorDB::new("test", TextEmbeddingAda002)
            .await
            .unwrap_err();
        assert!(matches!(result, PgVectorError::EnvVarError(_)));
    }

    #[tokio::test]
    async fn test_throws_error_when_cant_connect() {
        std::env::set_var("POSTGRES_USER", "postgres");
        std::env::set_var("POSTGRES_PASSWORD", "postgres");
        std::env::set_var("POSTGRES_HOST", "localhost");
        std::env::set_var("POSTGRES_DATABASE", "postgres");
        let result = PgVectorDB::new("test", TextEmbeddingAda002)
            .await
            .unwrap_err();
        assert!(matches!(result, PgVectorError::ConnectionError(_)));
    }
}
