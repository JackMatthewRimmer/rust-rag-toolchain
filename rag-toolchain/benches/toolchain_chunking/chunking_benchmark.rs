use std::fs;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rag_toolchain::toolchain_chunking::chunker::generate_chunks;
/*
Benchmarking of 30,000 words
*/
pub fn benchmark_chunking(c: &mut Criterion) {
  c.bench_function("chunking", |b| b.iter(|| {
    // i need to load the raw.txt file here
    let raw_text = fs::read_to_string("benches/raw.txt").unwrap() ;
    let window_size = 1200;
    let chunk_size = 8000;
    let chunks = generate_chunks(black_box(raw_text.as_str()), black_box(window_size), black_box(chunk_size));
    assert!(chunks.is_ok());
  }));
}

criterion_group!(benches, benchmark_chunking);
criterion_main!(benches);