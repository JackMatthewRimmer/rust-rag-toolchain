use crate::toolchain_indexing::traits::{EmbeddingStore, StoreError};
use async_trait::async_trait;
use sqlx::postgres::{PgPoolOptions, PgQueryResult};
use sqlx::Error as SqlxError;
use sqlx::{Pool, Postgres};
use std::env::{self, VarError};
use std::error::Error;
use std::fmt::{Display, Formatter};

use dotenv::dotenv;

#[derive(Debug)]
pub enum PgVectorError {
    EnvVarError(VarError),
    ConnectionError(SqlxError),
    TableCreationError(SqlxError),
    RuntimeCreationError(String),
    UpsertError(SqlxError),
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
            PgVectorError::RuntimeCreationError(error) => {
                write!(f, "Runtime creation error: {}", error)
            }
            PgVectorError::UpsertError(error) => {
                write!(f, "Upsert error: {}", error)
            }
            PgVectorError::TransactionError(error) => {
                write!(f, "Transaction error: {}", error)
            }
        }
    }
}

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
    table_name: String,
    pub pool: Pool<Postgres>,
}

impl PgVectorDB {
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
    pub async fn new(table_name: &str) -> Result<Self, PgVectorError> {
        dotenv().ok();
        let username: String = env::var("POSTGRES_USER")?;
        let password: String = env::var("POSTGRES_PASSWORD")?;
        let host: String = env::var("POSTGRES_HOST")?;
        let db_name: String = env::var("POSTGRES_DATABASE")?;
        let table_name: &str = table_name;

        let connection_string =
            format!("postgres://{}:{}@{}/{}", username, password, host, db_name);

        // Connect to the database
        let pool = PgVectorDB::connect(&connection_string)
            .await
            .map_err(PgVectorError::ConnectionError)?;

        // Create the table
        PgVectorDB::create_table(pool.clone(), table_name)
            .await
            .map_err(PgVectorError::TableCreationError)?;

        Ok(PgVectorDB {
            table_name: table_name.into(),
            pool,
        })
    }

    /// Allows us to check the connection to a database and store the connection pool
    ///
    /// # Arguments
    /// * `connection_string` - The connection string to use to connect to the database
    /// * `rt` - The runtime to use to connect to the database
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

    /// We call the create table automatically when the struct is created
    ///
    /// # Arguments
    /// * `table_name` - The name of the table to create
    /// * `pool` - The connection pool to use to create the table
    /// * `rt` - The runtime to use to create the table
    ///
    /// # Returns
    /// * [`PgQueryResult`] which can be used to check if the table was created successfully
    /// * [`Error`] if the table could not be created
    async fn create_table(
        pool: Pool<Postgres>,
        table_name: &str,
    ) -> Result<PgQueryResult, SqlxError> {
        let query = format!(
            "
            CREATE TABLE IF NOT EXISTS {} (
                id SERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                embedding vector(1536) 
            )",
            table_name
        );
        sqlx::query(&query).execute(&pool).await
    }
}

#[async_trait]
impl EmbeddingStore for PgVectorDB {
    async fn store(&self, embeddings: (String, Vec<f32>)) -> Result<(), Box<dyn Error>> {
        let (content, embedding) = embeddings;
        let query = format!(
            "
            INSERT INTO {} (content, embedding) VALUES ($1, $2::vector)",
            &self.table_name
        );

        sqlx::query(&query)
            .bind(&content)
            .bind(&embedding)
            .execute(&self.pool)
            .await
            .map_err(PgVectorError::UpsertError)?;
        Ok(())
    }

    async fn store_batch(&self, embeddings: Vec<(String, Vec<f32>)>) -> Result<(), Box<dyn Error>> {
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
            sqlx::query(&query)
                .bind(&content)
                .bind(&embedding)
                .execute(&mut *transaction)
                .await
                .map_err(PgVectorError::UpsertError)?;
        }
        transaction
            .commit()
            .await
            .map_err(PgVectorError::TransactionError)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_throws_env_var_error_user() {
        let result = PgVectorDB::new("test").await.unwrap_err();
        assert!(matches!(result, PgVectorError::EnvVarError(_)));
    }

    #[tokio::test]
    async fn test_throws_env_var_error_password() {
        std::env::set_var("POSTGRES_USER", "postgres");
        let result = PgVectorDB::new("test").await.unwrap_err();
        assert!(matches!(result, PgVectorError::EnvVarError(_)));
    }

    #[tokio::test]
    async fn test_throws_env_var_error_host() {
        std::env::set_var("POSTGRES_USER", "postgres");
        std::env::set_var("POSTGRES_PASSWORD", "postgres");
        let result = PgVectorDB::new("test").await.unwrap_err();

        assert!(matches!(result, PgVectorError::EnvVarError(_)));
    }

    #[tokio::test]
    async fn test_throws_env_var_error_database() {
        std::env::set_var("POSTGRES_USER", "postgres");
        std::env::set_var("POSTGRES_PASSWORD", "postgres");
        std::env::set_var("POSTGRES_HOST", "localhost");
        let result = PgVectorDB::new("test").await.unwrap_err();
        assert!(matches!(result, PgVectorError::EnvVarError(_)));
    }

    #[tokio::test]
    async fn test_throws_error_when_cant_connect() {
        std::env::set_var("POSTGRES_USER", "postgres");
        std::env::set_var("POSTGRES_PASSWORD", "postgres");
        std::env::set_var("POSTGRES_HOST", "localhost");
        std::env::set_var("POSTGRES_DATABASE", "postgres");
        let result = PgVectorDB::new("test").await.unwrap_err();
        assert!(matches!(result, PgVectorError::ConnectionError(_)));
    }
}
