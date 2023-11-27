use crate::toolchain_indexing::traits::EmbeddingStore;
use sqlx::postgres::{PgPoolOptions, PgQueryResult};
use sqlx::Error as SqlxError;
use sqlx::{Error, Pool, Postgres};
use std::env::{self, VarError};
use tokio::runtime::Runtime;

use dotenv::dotenv;

#[derive(Debug)]
pub enum PgVectorError {
    EnvVarError(VarError),
    ConnectionError(SqlxError),
    TableCreationError(SqlxError),
    RuntimeCreationError(String),
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
    rt: Runtime,
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
    pub fn new(table_name: &str) -> Result<Self, PgVectorError> {
        dotenv().ok();
        let username: String = env::var("POSTGRES_USER").map_err(PgVectorError::EnvVarError)?;
        let password: String = env::var("POSTGRES_PASSWORD").map_err(PgVectorError::EnvVarError)?;
        let host: String = env::var("POSTGRES_HOST").map_err(PgVectorError::EnvVarError)?;
        let db_name: String = env::var("POSTGRES_DATABASE").map_err(PgVectorError::EnvVarError)?;
        let table_name: &str = table_name;

        let connection_string =
            format!("postgres://{}:{}@{}/{}", username, password, host, db_name);

        let rt = tokio::runtime::Runtime::new()
            .map_err(|error| PgVectorError::RuntimeCreationError(error.to_string()))?;

        // Connect to the database
        let pool =
            PgVectorDB::connect(&connection_string, &rt).map_err(PgVectorError::ConnectionError)?;

        // Create the table
        PgVectorDB::create_table(&rt, pool.clone(), table_name)
            .map_err(PgVectorError::TableCreationError)?;

        Ok(PgVectorDB {
            table_name: table_name.into(),
            pool,
            rt,
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
    fn connect(connection_string: &str, rt: &Runtime) -> Result<Pool<Postgres>, Error> {
        let pool = rt.block_on(async {
            let pool: Pool<Postgres> = PgPoolOptions::new()
                .max_connections(5)
                .connect(connection_string)
                .await
                .expect("Error: Could not create pool to database");
            pool
        });
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
    fn create_table(
        rt: &Runtime,
        pool: Pool<Postgres>,
        table_name: &str,
    ) -> Result<PgQueryResult, Error> {
        let query = format!(
            "
            CREATE TABLE IF NOT EXISTS {} (
                id SERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                embedding vector(1536) 
            )",
            table_name
        );
        rt.block_on(async { sqlx::query(&query).execute(&pool).await })
    }
}

impl EmbeddingStore for PgVectorDB {
    fn store(&self, embeddings: (String, Vec<f32>)) -> Result<(), std::io::Error> {
        let (content, embedding) = embeddings;
        let query = format!(
            "
            INSERT INTO {} (content, embedding) VALUES ($1, $2::vector)",
            &self.table_name
        );

        let result = self.rt.block_on(async {
            sqlx::query(&query)
                .bind(&content)
                .bind(&embedding)
                .execute(&self.pool)
                .await
        });

        match result {
            Ok(_) => Ok(()),
            Err(_) => Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Error: Could not store embedding",
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_throws_env_var_error_user() {
        let result = PgVectorDB::new("test");
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), PgVectorError::EnvVarError(_)));
    }

    #[test]
    fn test_throws_env_var_error_password() {
        std::env::set_var("POSTGRES_USER", "postgres");
        let result = PgVectorDB::new("test");
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), PgVectorError::EnvVarError(_)));
    }

    #[test]
    fn test_throws_env_var_error_host() {
        std::env::set_var("POSTGRES_USER", "postgres");
        std::env::set_var("POSTGRES_PASSWORD", "postgres");
        let result = PgVectorDB::new("test");
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), PgVectorError::EnvVarError(_)));
    }

    #[test]
    fn test_throws_env_var_error_database() {
        std::env::set_var("POSTGRES_USER", "postgres");
        std::env::set_var("POSTGRES_PASSWORD", "postgres");
        std::env::set_var("POSTGRES_HOST", "localhost");
        let result = PgVectorDB::new("test");
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), PgVectorError::EnvVarError(_)));
    }
}
