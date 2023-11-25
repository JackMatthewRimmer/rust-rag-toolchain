use rag_toolchain::toolchain_indexing::stores::pg_vector_store::PgVector;
fn main() {
    let pg_vector = PgVector::new("name").unwrap();
}
