use crate::toolchain_indexing::traits::EmbeddingStore;
use sqlx::postgres::{PgPoolOptions, PgQueryResult};
use sqlx::types::Json;
use sqlx::{Error, Pool, Postgres};
use std::env;
use tokio::runtime::Runtime;

use dotenv::dotenv;

#[derive(Debug)]
pub enum PgVectorError {
    EnvVarError(String),
    ConnectionError(String),
    TableCreationError(String),
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
pub struct PgVector {
    table_name: String,
    connection_string: String,
    pub pool: Pool<Postgres>,
    rt: Runtime,
}

impl PgVector {
    /// # Note
    /// [`PgVector::new`] will not compile if the required environment
    /// variables are not set
    ///
    /// # Arguments
    /// * `db_name` - The name of the table to store the embeddings in.
    ///
    /// # Returns
    /// the constructed [`PgVector`] struct
    pub fn new(table_name: &str) -> Result<Self, PgVectorError> {
        dotenv().ok();
        let username = env::var("POSTGRES_USER")
            .map_err(|_| PgVectorError::EnvVarError("Error: POSTGRES_USER not set".into()))?;
        let password = env::var("POSTGRES_PASSWORD")
            .map_err(|_| PgVectorError::EnvVarError("Error: POSTGRES_PASSWORD not set".into()))?;
        let host = env::var("POSTGRES_HOST")
            .map_err(|_| PgVectorError::EnvVarError("Error: POSTGRES_HOST not set".into()))?;
        let db_name = env::var("POSTGRES_DATABASE")
            .map_err(|_| PgVectorError::EnvVarError("Error: POSTGRES_DATABASE not set".into()))?;
        let table_name = table_name.into();

        let connection_string =
            format!("postgres://{}:{}@{}/{}", username, password, host, db_name);

        // Better error handling here
        let rt = tokio::runtime::Runtime::new().unwrap();

        let pool = PgVector::connect(&connection_string, &rt).map_err(|_| {
            PgVectorError::ConnectionError("Error: Could not connect to database".into())
        })?;

        Ok(PgVector {
            table_name,
            connection_string,
            pool,
            rt,
        })
    }

    fn connect(connection_string: &str, rt: &Runtime) -> Result<Pool<Postgres>, Error> {
        let pool = rt.block_on(async {
            let pool: Pool<Postgres> = PgPoolOptions::new()
                .max_connections(5)
                .connect(&connection_string)
                .await
                .expect("Error: Could not create pool to database");
            return pool;
        });
        Ok(pool)
    }

    /// # Create Table
    /// This operation should be called if your table does not already exist.
    /// Will fail if the table already exists.
    pub fn create_table(&self) -> Result<PgQueryResult, Error> {
        let query = format!(
            "
            CREATE TABLE IF NOT EXISTS {} (
                id SERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                embedding vector(1536) 
            )",
            &self.table_name
        );
        return self.rt.block_on(async {
            return sqlx::query(&query).execute(&self.pool).await;
        });
    }
}

impl EmbeddingStore for PgVector {
    fn store(&self, embeddings: (String, Vec<f32>)) -> Result<(), std::io::Error> {
        let (content, embedding) = embeddings;
        let query = format!(
            "
            INSERT INTO {} (content, embedding) VALUES ($1, $2::vector)",
            &self.table_name
        );

        let result = self.rt.block_on(async {
            return sqlx::query(&query)
                .bind(&content)
                .bind(&embedding)
                .execute(&self.pool)
                .await;
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
