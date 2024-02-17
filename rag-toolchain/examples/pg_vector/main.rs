use rag_toolchain::chunkers::TokenChunker;
use rag_toolchain::clients::{AsyncEmbeddingClient, OpenAIEmbeddingClient};
use rag_toolchain::common::{Chunk, Chunks, Embedding, OpenAIEmbeddingModel};
use rag_toolchain::stores::{DistanceFunction, EmbeddingStore, PostgresVectorStore, HNSW};

#[tokio::main]
async fn main() {
    std::env::set_var("POSTGRES_USER", "postgres");
    std::env::set_var("POSTGRES_PASSWORD", "postgres");
    std::env::set_var("POSTGRES_HOST", "localhost");
    std::env::set_var("POSTGRES_DATABASE", "postgres");

    // Given our text, we want to split it into chunks, Usually this would be done for larger texts
    // But has been done here for demonstration purposes
    let text = std::fs::read_to_string("examples/pg_vector/example_text.txt").unwrap();
    println!("Text: {}", text);
    let chunker = TokenChunker::try_new(
        std::num::NonZeroUsize::new(50).unwrap(),
        25,
        OpenAIEmbeddingModel::TextEmbeddingAda002,
    )
    .unwrap();
    let chunks: Chunks = chunker.generate_chunks(&text).unwrap();
    println!("Chunks: {:?}", chunks);

    // I would check your store initialized before sending of embeddings to openai
    let store: PostgresVectorStore<HNSW> = PostgresVectorStore::<HNSW>::try_new(
        "embeddings",
        OpenAIEmbeddingModel::TextEmbeddingAda002,
        DistanceFunction::Cosine,
    )
    .await
    .unwrap();

    // Create a new client and generate the embeddings for the chunks
    let client: OpenAIEmbeddingClient =
        OpenAIEmbeddingClient::try_new(OpenAIEmbeddingModel::TextEmbeddingAda002).unwrap();
    let embeddings: Vec<(Chunk, Embedding)> = client.generate_embeddings(chunks).await.unwrap();
    println!("Embeddings: {:?}", embeddings);

    // Insert the embeddings into the store this will be a better interface in the future
    store.store_batch(embeddings).await.unwrap();
}
