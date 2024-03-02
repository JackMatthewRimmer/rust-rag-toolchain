use pgvector::Vector;

use crate::retrievers::DistanceFunction;

/// # [`PostgresRow`]
/// Type that represents a row in our defined structure
/// for postgres store and postgres retriever.
#[derive(Debug, Clone, PartialEq, sqlx::FromRow)]
pub struct PostgresRow {
    id: i32,
    content: String,
    embedding: Vector,
    #[sqlx(json)]
    metadata: serde_json::Value,
}

fn create_table_sql(table_name: &str) -> String {
    format!(
        "CREATE TABLE IF NOT EXISTS {} (
        id SERIAL PRIMARY KEY,
        content TEXT NOT NULL,
        embedding VECTOR NOT NULL,
        metadata JSONB)",
        table_name
    )
}

fn upsert_row_sql(table_name: &str) -> String {
    format!("INSERT INTO {} (id, content, embedding, metadata) VALUES ($1, $2, $3, $4)
    ON CONFLICT (id) DO UPDATE SET content = EXCLUDED.content, embedding = EXCLUDED.embedding, metadata = EXCLUDED.metadata", table_name)
}

fn select_row_sql(table_name: &str, distance_function: DistanceFunction) -> String {
    format!(
        "SELECT id, content, embedding, metadata FROM {} ORDER BY embedding {} $1::vector LIMIT $2",
        table_name,
        distance_function.to_sql_string()
    )
}