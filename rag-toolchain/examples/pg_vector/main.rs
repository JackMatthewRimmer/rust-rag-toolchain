use rag_toolchain::chunkers::{Chunker, TokenChunker};
use rag_toolchain::clients::{AsyncEmbeddingClient, OpenAIEmbeddingClient};
use rag_toolchain::common::{Chunks, Embedding, OpenAIEmbeddingModel};
use rag_toolchain::stores::{EmbeddingStore, PostgresVectorStore};

#[tokio::main]
async fn main() {
    const EMBEDDING_MODEL: OpenAIEmbeddingModel = OpenAIEmbeddingModel::TextEmbeddingAda002;

    // We read in the text from a file
    let text = std::fs::read_to_string("examples/pg_vector/example_text.txt").unwrap();

    // Create a new chunker and generate the chunks
    let chunker = TokenChunker::try_new(
        std::num::NonZeroUsize::new(50).unwrap(),
        25,
        EMBEDDING_MODEL,
    )
    .unwrap();
    let chunks: Chunks = chunker.generate_chunks(&text).unwrap();

    // I would check your store initialized before sending of embeddings to openai
    let store: PostgresVectorStore = PostgresVectorStore::try_new("embeddings", EMBEDDING_MODEL)
        .await
        .unwrap();

    // Create a new client and generate the embeddings for the chunks
    let client: OpenAIEmbeddingClient = OpenAIEmbeddingClient::try_new(EMBEDDING_MODEL).unwrap();
    let embeddings: Vec<Embedding> = client.generate_embeddings(chunks).await.unwrap();

    // Insert the embeddings into the store
    store.store_batch(embeddings).await.unwrap();
}
