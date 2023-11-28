use pgvector::Vector;
use rag_toolchain::toolchain_indexing::stores::pg_vector::PgVectorDB;
use rag_toolchain::toolchain_indexing::traits::EmbeddingStore;
use sqlx::{postgres::PgRow, Row};
use sqlx::{Pool, Postgres};
use tokio::runtime::Runtime;

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
        assert_row(&pg_vector.pool, 1, "test".into(), vec![1.0; 1536])
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
        assert_row(&pg_vector.pool, 1, "test1".into(), vec![1.0; 1536]);
        assert_row(&pg_vector.pool, 2, "test2".into(), vec![2.0; 1536]);
        assert_row(&pg_vector.pool, 3, "test3".into(), vec![3.0; 1536]);
    }

    fn assert_row(pool: &Pool<Postgres>, id: i32, text: String, embeddings: Vec<f32>) -> () {
        let rt: Runtime = tokio::runtime::Runtime::new().unwrap();
        let row: PgRow = rt.block_on(async {
            let query: Result<PgRow, sqlx::Error> =
                sqlx::query("SELECT id, content, embedding FROM embeddings WHERE id = $1")
                    .bind(id)
                    .fetch_one(pool)
                    .await;
            return query.unwrap();
        });

        assert_eq!(row.get::<String, _>("content"), text);
        assert_eq!(row.get::<Vector, _>("embedding").to_vec(), embeddings);
    }
}
