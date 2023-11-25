use rag_toolchain::toolchain_indexing::stores::pg_vector_store::PgVector;
use rag_toolchain::toolchain_indexing::traits::EmbeddingStore;

#[cfg(test)]
mod pg_vector {

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
    }
}
