use pgvector::Vector;
use rag_toolchain::toolchain_indexing::stores::pg_vector::PgVectorDB;
use rag_toolchain::toolchain_indexing::traits::EmbeddingStore;
use sqlx::{postgres::PgRow, Row};
use sqlx::{Pool, Postgres};

#[cfg(test)]
mod pg_vector {

    use super::*;

    #[tokio::test]
    async fn test_store_persists() {
        std::env::set_var("POSTGRES_USER", "postgres");
        std::env::set_var("POSTGRES_PASSWORD", "postgres");
        std::env::set_var("POSTGRES_HOST", "localhost");
        std::env::set_var("POSTGRES_DATABASE", "pg_vector");
        let pg_vector = PgVectorDB::new("test_db_1").await.unwrap();
        let _result = pg_vector.store(("test".into(), vec![1.0; 1536])).await.map_err(|_| panic!("panic"));
        assert_row(&pg_vector.pool, 1, "test".into(), vec![1.0; 1536]).await;
    }

    #[tokio::test]
    async fn test_batch_store_persists() {
        std::env::set_var("POSTGRES_USER", "postgres");
        std::env::set_var("POSTGRES_PASSWORD", "postgres");
        std::env::set_var("POSTGRES_HOST", "localhost");
        std::env::set_var("POSTGRES_DATABASE", "pg_vector");
        let pg_vector = PgVectorDB::new("test_db_2").await.unwrap();
        let input: Vec<(String, Vec<f32>)> = vec![
            ("test1".into(), vec![1.0; 1536]),
            ("test2".into(), vec![2.0; 1536]),
            ("test3".into(), vec![3.0; 1536]),
        ];
        let _result = pg_vector.store_batch(input).await.map_err(|_| panic!("panic"));
        assert_row(&pg_vector.pool, 1, "test1".into(), vec![1.0; 1536]).await;
        assert_row(&pg_vector.pool, 2, "test2".into(), vec![2.0; 1536]).await;
        assert_row(&pg_vector.pool, 3, "test3".into(), vec![3.0; 1536]).await;
    }

    async fn assert_row(pool: &Pool<Postgres>, id: i32, text: String, embeddings: Vec<f32>) -> () {
        let row: RowData = query_row(pool, id).await;
        assert_eq!(row.id, id);
        assert_eq!(row.content, text);
        assert_eq!(row.embedding, embeddings);
    }

    async fn query_row(pool: &Pool<Postgres>, id: i32) -> RowData {
        let query: PgRow =
            sqlx::query("SELECT id, content, embedding FROM embeddings WHERE id = $1")
                .bind(id)
                .fetch_one(pool)
                .await
                .unwrap();
        RowData {
            id: query.get::<i32, _>("id"),
            content: query.get::<String, _>("content"),
            embedding: query.get::<Vector, _>("embedding").to_vec(),
        }
    }

    struct RowData {
        id: i32,
        content: String,
        embedding: Vec<f32>,
    }
}
