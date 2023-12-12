use std::fs;
use std::num::NonZeroUsize;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rag_toolchain::chunkers::token_chunker::TokenChunker;
use rag_toolchain::common::embedding_shared::OpenAIEmbeddingModel::TextEmbeddingAda002;
/*
Benchmarking of 30,000 words
*/
pub fn benchmark_chunking(c: &mut Criterion) {
    c.bench_function("chunking", |b| {
        b.iter(|| {
            // i need to load the raw.txt file here
            let raw_text = fs::read_to_string("benches/raw.txt").unwrap();
            let window_size = 1200;
            let chunk_size = NonZeroUsize::new(8000).unwrap();
            let chunker: TokenChunker = TokenChunker::new(
                black_box(chunk_size),
                black_box(window_size),
                black_box(TextEmbeddingAda002),
            )
            .unwrap();
            let chunks = chunker.generate_chunks(&raw_text);
            assert!(chunks.is_ok());
        })
    });
}

criterion_group!(benches, benchmark_chunking);
criterion_main!(benches);
