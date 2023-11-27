use rag_toolchain::toolchain_indexing::stores::pg_vector::PgVector;
use rag_toolchain::toolchain_indexing::traits::EmbeddingStore;
use sqlx::{Pool, Postgres};

#[cfg(test)]
mod pg_vector {

    use sqlx::{postgres::PgRow, Row};
    use tokio::runtime::Runtime;

    use super::*;

    #[test]
    fn check() {
        std::env::set_var("POSTGRES_USER", "postgres");
        std::env::set_var("POSTGRES_PASSWORD", "postgres");
        std::env::set_var("POSTGRES_HOST", "localhost");
        std::env::set_var("POSTGRES_DATABASE", "pg_vector");
        let pg_vector = PgVector::new("embeddings").unwrap();
        let _result = pg_vector.create_table().unwrap();
        let _result = pg_vector.store(("test".into(), vec![1.0; 1536])).unwrap();
        assert_row(pg_vector.pool, 1, "test".into(), vec![1.0; 1536])
    }

    fn assert_row(pool: Pool<Postgres>, id: i32, text: String, embeddings: Vec<f32>) -> () {
        let rt: Runtime = tokio::runtime::Runtime::new().unwrap();
        let row: PgRow = rt.block_on(async {
            let query: Result<PgRow, sqlx::Error> =
                sqlx::query("SELECT id, content, embedding FROM embeddings WHERE id = $1")
                    .bind(id)
                    .fetch_one(&pool)
                    .await;
            return query.unwrap();
        });

        assert_eq!(row.get::<String, _>("content"), "test");
    }
}
