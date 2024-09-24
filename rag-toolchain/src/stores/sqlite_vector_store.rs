use thiserror::Error;

use crate::common::EmbeddingModel;

use super::EmbeddingStore;

#[derive(Debug)]
pub struct SqliteVectorStore {
    conn: rusqlite::Connection,
    table_name: String,
}

impl SqliteVectorStore {
    pub fn try_new_with_connection(
        conn: rusqlite::Connection,
        table_name: &str,
        embedding_model: impl EmbeddingModel,
    ) -> Result<Self, SqliteVectorStoreError> {
        let vector_dimension = embedding_model.metadata().dimensions;
        SqliteVectorStore::create_table(&conn, table_name, vector_dimension)?;
        let store: SqliteVectorStore = SqliteVectorStore {
            conn,
            table_name: table_name.into(),
        };

        Ok(store)
    }

    fn create_table(
        conn: &rusqlite::Connection,
        table_name: &str,
        vector_dimension: usize,
    ) -> Result<(), SqliteVectorStoreError> {
        let stmt: String = format!(
            "CREATE VIRTUAL TABLE IF NOT EXISTS {} using vec0(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                embedding float[{}] NOT NULL,
                metadata TEXT NOT NULL,
            )",
            table_name, vector_dimension
        );

        conn.execute(&stmt, [])
            .map(|_| ())
            .map_err(SqliteVectorStoreError::TableCreationError)
    }
}

impl EmbeddingStore for SqliteVectorStore {
    type ErrorType = SqliteVectorStoreError;

    async fn store(&self, embedding: crate::common::Embedding) -> Result<(), Self::ErrorType> {}

    async fn store_batch(
        &self,
        embeddings: Vec<crate::common::Embedding>,
    ) -> Result<(), Self::ErrorType> {
    }
}

#[derive(Error, Debug)]
pub enum SqliteVectorStoreError {
    #[error("Table Creation Error: {0}")]
    TableCreationError(rusqlite::Error),
}
