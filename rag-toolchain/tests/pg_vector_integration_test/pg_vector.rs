use rag_toolchain::toolchain_indexing::destinations::PgVector;
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

fn start_db() {
    println!("Starting test database...");
    let _output = Command::new("docker-compose")
        .current_dir("tests/pg_vector_integration_test/docker")
        .args(&["up", "-d"])
        .output()
        .expect("Failed to launch test db via docker-compose");
    println!("Database started");
}

fn stop_db() {
    println!("Stopping test database...");
    let _output = Command::new("docker-compose")
        .current_dir("tests/pg_vector_integration_test/docker")
        .args(&["down", "--rmi", "all"])
        .output()
        .expect("Failed to stop test db via docker-compose");
    println!("Database stopped");
}

#[cfg(test)]
mod pg_vector {

    use super::*;

    #[test]
    fn check() {
        start_db();

        std::env::set_var("POSTGRES_USERNAME", "postgres");
        std::env::set_var("POSTGRES_PASSWORD", "password");
        std::env::set_var("POSTGRES_HOST", "localhost");
        std::env::set_var("POSTGRES_DB", "pg_vector");

        let pg_vector = PgVector::new("embeddings").unwrap();

        stop_db();
    }
}
