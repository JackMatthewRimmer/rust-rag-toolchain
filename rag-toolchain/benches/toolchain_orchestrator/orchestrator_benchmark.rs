use std::{fs, future::IntoFuture};

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rag_toolchain::toolchain_orchestrator::orchestrator::generate_embeddings;

pub fn benchmark_orchestrating(c: &mut Criterion) {
  c.bench_function("orchestrating", |b| b.iter(|| {
    // i need to load the raw.txt file here
    let raw_text = fs::read_to_string("benches/raw.txt").unwrap() ;
    let input = vec![raw_text; 1].into_iter();
    let window_size = 1200;
    let chunk_size = 8000;
    let chunks = generate_embeddings(black_box(input), black_box(|x| Ok(())));
  }));
}

criterion_group!(benches, benchmark_orchestrating);
criterion_main!(benches);