use rag_toolchain::toolchain_embeddings::{
    embedding_models::OpenAIEmbeddingModel, openai_embeddings::OpenAIClient,
};
use rag_toolchain::toolchain_indexing::chunking::TokenChunker;
use rag_toolchain::toolchain_indexing::stores::pg_vector::PgVectorDB;
use rag_toolchain::toolchain_indexing::traits::{Chunks, EmbeddingStore};

#[tokio::main]
async fn main() {
    std::env::set_var("POSTGRES_USER", "postgres");
    std::env::set_var("POSTGRES_PASSWORD", "postgres");
    std::env::set_var("POSTGRES_HOST", "localhost");
    std::env::set_var("POSTGRES_DATABASE", "pg_vector");

    // Given our text, we want to split it into chunks, Usually this would be done for larger texts
    // But has been done here for demonstration purposes
    let text = std::fs::read_to_string("examples/pg_vector/example_text.txt").unwrap();
    println!("Text: {}", text);
    let chunker = TokenChunker::new(
        std::num::NonZeroUsize::new(50).unwrap(),
        25,
        OpenAIEmbeddingModel::TextEmbeddingAda002,
    )
    .unwrap();
    let chunks: Chunks = chunker.generate_chunks(&text).unwrap();
    println!("Chunks: {:?}", chunks);

    // I would check your store initialized before sending of embeddings to openai
    let store: PgVectorDB =
        PgVectorDB::new("embeddings", OpenAIEmbeddingModel::TextEmbeddingAda002)
            .await
            .unwrap();

    // Create a new client and generate the embeddings for the chunks
    let client: OpenAIClient = OpenAIClient::new().unwrap();
    let embeddings: Vec<(String, Vec<f32>)> = client.generate_embeddings(chunks).unwrap();
    println!("Embeddings: {:?}", embeddings);

    // Insert the embeddings into the store this will be a better interface in the future
    store.store_batch(embeddings).await.unwrap();
}
