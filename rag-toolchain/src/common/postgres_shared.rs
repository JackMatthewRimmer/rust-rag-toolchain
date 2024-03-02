use pgvector::Vector;
use sqlx::postgres::PgQueryResult;

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

fn select_row_sql(table_name: &str, distance_function: DistanceFunction) -> String {
    format!(
        "SELECT id, content, embedding, metadata FROM {} ORDER BY embedding {} $1::vector LIMIT $2",
        table_name,
        distance_function.to_sql_string()
    )
}
