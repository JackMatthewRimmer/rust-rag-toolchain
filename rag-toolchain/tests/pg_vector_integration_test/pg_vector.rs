use rag_toolchain::toolchain_indexing::destinations::PgVector;
use rag_toolchain::toolchain_indexing::traits::EmbeddingStore;
use std::process::Command;

// In order to run this test, you must have a Postgres database running on your machine
// Just run the docker compose file in the docker directory

// #!/bin/bash

// # Start the test database
// docker-compose up -d

// # Run the integration tests
// cargo test --test integration_tests

// # Stop the test database and remove the Docker image
// docker-compose down --rmi all

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
