use crate::clients::AsyncEmbeddingClient;
use crate::common::{Chunk, Embedding, EmbeddingModel};
use crate::retrievers::PostgresVectorRetriever;
use crate::stores::traits::EmbeddingStore;
use async_trait::async_trait;
use sqlx::postgres::{PgPoolOptions, PgQueryResult};
use sqlx::Error as SqlxError;
use sqlx::{Pool, Postgres};
use std::env::{self, VarError};
use std::error::Error;
use std::fmt::{Display, Formatter};

pub trait IndexTypes: Send + Sync {}
#[derive(Debug, Clone)]
pub struct IVFFLAT {
    distance_function: DistanceFunction,
}
impl IndexTypes for IVFFLAT {}
#[derive(Debug, Clone)]
pub struct HNSW {
    distance_function: DistanceFunction,
}
impl IndexTypes for HNSW {}
#[derive(Debug, Clone)]
pub struct NoIndex {}
impl IndexTypes for NoIndex {}

use dotenv::dotenv;

/// # PostgresVectorStore
///
/// This struct is used to store and as a retriever information needed to connect to a Postgres database
/// and should be passed to an embedding task as a destination for the data to be stored.
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
#[derive(Debug, Clone)]
pub struct PostgresVectorStore<T>
where
    T: IndexTypes,
{
    /// We make the pool public incase users want to
    /// do extra operations on the database
    pool: Pool<Postgres>,
    table_name: String,
    index_type: T,
}

impl<T> PostgresVectorStore<T>
where
    T: IndexTypes,
{
    pub fn get_pool(&self) -> Pool<Postgres> {
        self.pool.clone()
    }

    async fn connect_and_create_table(
        table_name: &str,
        embedding_model: impl EmbeddingModel,
    ) -> Result<Pool<Postgres>, PostgresVectorError> {
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
        let pool = Self::connect(&connection_string)
            .await
            .map_err(PostgresVectorError::ConnectionError)?;

        // Create the table
        Self::create_table(pool.clone(), table_name, embedding_diminsions)
            .await
            .map_err(PostgresVectorError::TableCreationError)?;

        Ok(pool)
    }

    /// # connect
    /// Allows us to check the connection to a database and store the connection pool
    ///
    /// # Arguments
    /// * `connection_string` - The connection string to use to connect to the database
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
impl<T> EmbeddingStore for PostgresVectorStore<T>
where
    T: IndexTypes,
{
    type ErrorType = PostgresVectorError;
    /// # store
    ///
    /// # Arguments
    /// * `embeddings` - A tuple containing the content and the embedding to store
    ///
    /// # Errors
    /// * [`PostgresVectorError::InsertError`] if the insert fails
    ///
    /// # Returns
    /// * [`()`] if the insert succeeds
    async fn store(&self, embeddings: (Chunk, Embedding)) -> Result<(), PostgresVectorError> {
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
            .map_err(PostgresVectorError::InsertError)?;
        Ok(())
    }

    /// # store_batch
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
        let query = format!(
            "
            INSERT INTO {} (content, embedding) VALUES ($1, $2::vector)",
            &self.table_name
        );

        let mut transaction = self
            .pool
            .begin()
            .await
            .map_err(PostgresVectorError::TransactionError)?;

        for (content, embedding) in embeddings {
            let text: String = content.into();
            let vector: Vec<f32> = embedding.into();
            sqlx::query(&query)
                .bind(text)
                .bind(vector)
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

impl PostgresVectorStore<NoIndex> {
    /// # try_new
    ///
    /// This function is used to create a new instance of the PostgresVectorStore
    ///
    /// # Arguments
    /// * `table_name` - The name of the table to store the embeddings in
    /// * `embedding_model` - The embedding model to use to store the embeddings
    ///
    /// # Errors
    /// * [`PostgresVectorError::EnvVarError`] if the environment variables are not set
    /// * [`PostgresVectorError::ConnectionError`] if the connection to the database could not be established
    /// * [`PostgresVectorError::TableCreationError`] if the table could not be created
    ///
    /// # Returns
    /// * [`PostgresVectorStore`] if the store is created successfully
    pub async fn try_new(
        table_name: &str,
        embedding_model: impl EmbeddingModel,
    ) -> Result<PostgresVectorStore<NoIndex>, PostgresVectorError> {
        let pool = Self::connect_and_create_table(table_name, embedding_model).await?;
        Ok(PostgresVectorStore {
            pool,
            table_name: table_name.to_string(),
            index_type: NoIndex {},
        })
    }

    pub fn as_retriever<G: AsyncEmbeddingClient>(
        &self,
        embedding_client: G,
        distance_function: DistanceFunction,
    ) -> PostgresVectorRetriever<G> {
        PostgresVectorRetriever::new(
            self.pool.clone(),
            self.table_name.clone(),
            embedding_client,
            distance_function,
        )
    }
}

impl PostgresVectorStore<HNSW> {
    /// # try_new
    ///
    /// This function is used to create a new instance of the PostgresVectorStore
    ///
    /// # Arguments
    /// * `table_name` - The name of the table to store the embeddings in
    /// * `embedding_model` - The embedding model to use to store the embeddings
    /// * `index_type` - The type of index to enable on the table
    ///
    /// # Errors
    /// * [`PostgresVectorError::EnvVarError`] if the environment variables are not set
    /// * [`PostgresVectorError::ConnectionError`] if the connection to the database could not be established
    /// * [`PostgresVectorError::TableCreationError`] if the table could not be created
    ///
    /// # Returns
    /// * [`PostgresVectorStore`] if the store is created successfully
    pub async fn try_new(
        table_name: &str,
        embedding_model: impl EmbeddingModel,
        distance_function: DistanceFunction,
    ) -> Result<PostgresVectorStore<HNSW>, PostgresVectorError> {
        let pool = Self::connect_and_create_table(table_name, embedding_model).await?;
        Self::enable_hnsw_index(&pool, table_name, distance_function.clone())
            .await
            .map_err(PostgresVectorError::TableCreationError)?;
        let store = PostgresVectorStore {
            pool,
            table_name: table_name.to_string(),
            index_type: HNSW { distance_function },
        };
        Ok(store)
    }

    pub fn as_retriever<G: AsyncEmbeddingClient>(
        &self,
        embedding_client: G,
    ) -> PostgresVectorRetriever<G> {
        PostgresVectorRetriever::new(
            self.pool.clone(),
            self.table_name.clone(),
            embedding_client,
            self.index_type.distance_function.clone(),
        )
    }

    async fn enable_hnsw_index(
        pool: &Pool<Postgres>,
        table_name: &str,
        distance_function: DistanceFunction,
    ) -> Result<PgQueryResult, SqlxError> {
        let query: String = format!(
            "
         CREATE INDEX ON {} USING hnsw (embedding {});
        ",
            table_name,
            distance_function.to_ddl_string()
        );
        sqlx::query(&query).execute(pool).await
    }
}

impl PostgresVectorStore<IVFFLAT> {
    /// # try_new
    ///
    /// This function is used to create a new instance of the PostgresVectorStore
    ///
    /// # Arguments
    /// * `table_name` - The name of the table to store the embeddings in
    /// * `embedding_model` - The embedding model to use to store the embeddings
    /// * `index_type` - The type of index to enable on the table
    ///
    /// # Errors
    /// * [`PostgresVectorError::EnvVarError`] if the environment variables are not set
    /// * [`PostgresVectorError::ConnectionError`] if the connection to the database could not be established
    /// * [`PostgresVectorError::TableCreationError`] if the table could not be created
    ///
    /// # Returns
    /// * [`PostgresVectorStore`] if the store is created successfully
    pub async fn try_new(
        table_name: &str,
        embedding_model: impl EmbeddingModel,
        distance_function: DistanceFunction,
        number_of_lists: u32,
    ) -> Result<PostgresVectorStore<IVFFLAT>, PostgresVectorError> {
        let pool = Self::connect_and_create_table(table_name, embedding_model).await?;
        Self::enable_ivfflat_index(
            &pool,
            table_name,
            distance_function.clone(),
            number_of_lists,
        )
        .await
        .map_err(PostgresVectorError::TableCreationError)?;
        let store = PostgresVectorStore {
            pool,
            table_name: table_name.to_string(),
            index_type: IVFFLAT { distance_function },
        };
        Ok(store)
    }

    pub fn as_retriever<G: AsyncEmbeddingClient>(
        &self,
        embedding_client: G,
    ) -> PostgresVectorRetriever<G> {
        PostgresVectorRetriever::new(
            self.pool.clone(),
            self.table_name.clone(),
            embedding_client,
            self.index_type.distance_function.clone(),
        )
    }

    async fn enable_ivfflat_index(
        pool: &Pool<Postgres>,
        table_name: &str,
        distance_function: DistanceFunction,
        number_of_lists: u32,
    ) -> Result<PgQueryResult, SqlxError> {
        let query: String = format!(
            "
         CREATE INDEX ON {} USING ivfflat (embedding {}) WITH (lists = {});
        ",
            table_name,
            distance_function.to_ddl_string(),
            number_of_lists
        );
        sqlx::query(&query).execute(pool).await
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DistanceFunction {
    L2,
    Cosine,
    InnerProduct,
}

impl DistanceFunction {
    pub fn to_ddl_string(&self) -> &str {
        match self {
            DistanceFunction::L2 => "vector_l2_ops",
            DistanceFunction::Cosine => "vector_cosine_ops",
            DistanceFunction::InnerProduct => "vector_ip_ops",
        }
    }

    pub fn to_sql_string(&self) -> &str {
        match self {
            DistanceFunction::L2 => "<->",
            DistanceFunction::Cosine => "<=>",
            DistanceFunction::InnerProduct => "<#>",
        }
    }
}

/// # PgVectorError
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
        let result = PostgresVectorStore::<NoIndex>::try_new("test", TextEmbeddingAda002)
            .await
            .unwrap_err();
        assert!(matches!(result, PostgresVectorError::EnvVarError(_)));

        std::env::set_var("POSTGRES_USER", "postgres");
        let result = PostgresVectorStore::<NoIndex>::try_new("test", TextEmbeddingAda002)
            .await
            .unwrap_err();
        assert!(matches!(result, PostgresVectorError::EnvVarError(_)));

        std::env::set_var("POSTGRES_PASSWORD", "postgres");
        let result = PostgresVectorStore::<NoIndex>::try_new("test", TextEmbeddingAda002)
            .await
            .unwrap_err();
        assert!(matches!(result, PostgresVectorError::EnvVarError(_)));

        std::env::set_var("POSTGRES_HOST", "localhost");
        let result = PostgresVectorStore::<NoIndex>::try_new("test", TextEmbeddingAda002)
            .await
            .unwrap_err();
        assert!(matches!(result, PostgresVectorError::EnvVarError(_)));

        std::env::set_var("POSTGRES_DATABASE", "postgres");
        let result = PostgresVectorStore::<NoIndex>::try_new("test", TextEmbeddingAda002)
            .await
            .unwrap_err();
        assert!(matches!(result, PostgresVectorError::ConnectionError(_)));
    }
}
