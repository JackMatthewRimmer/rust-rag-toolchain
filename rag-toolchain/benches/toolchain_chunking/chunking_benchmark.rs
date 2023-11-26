use std::fs;
use std::num::NonZeroUsize;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rag_toolchain::toolchain_embeddings::embedding_models::{
    EmbeddingModels, OpenAIEmbeddingModel, OpenAIEmbeddingModel::TextEmbeddingAda002,
};
use rag_toolchain::toolchain_indexing::chunking::TokenChunker;
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
            let model: EmbeddingModels = TextEmbeddingAda002.into();
            let chunker: TokenChunker = TokenChunker::new(
                black_box(chunk_size),
                black_box(window_size),
                black_box(model),
            )
            .unwrap();
            let chunks = chunker.generate_chunks(&raw_text);
            assert!(chunks.is_ok());
        })
    });
}

criterion_group!(benches, benchmark_chunking);
criterion_main!(benches);
