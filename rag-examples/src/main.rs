use rag_toolchain::toolchain_embeddings::{
    embedding_models::OpenAIEmbeddingModel, openai_embeddings::OpenAIClient,
};
use rag_toolchain::toolchain_indexing::chunking::TokenChunker;
use rag_toolchain::toolchain_indexing::stores::pg_vector::PgVector;
use rag_toolchain::toolchain_indexing::traits::EmbeddingStore;

fn main() {
    // Given our text, we want to split it into chunks, Usually this would be done for larger texts
    // But has been done here for demonstration purposes
    let text = std::fs::read_to_string("src/example_text.txt").unwrap();
    println!("Text: {}", text);
    let chunker = TokenChunker::new(
        std::num::NonZeroUsize::new(50).unwrap(),
        25,
        OpenAIEmbeddingModel::TextEmbeddingAda002.into(),
    )
    .unwrap();
    let chunks: Vec<String> = chunker.generate_chunks(&text).unwrap();
    println!("Chunks: {:?}", chunks);

    // I would check your store initialized before sending of embeddings to openai
    let store: PgVector = PgVector::new("embeddings").unwrap();

    // Create a new client and generate the embeddings for the chunks
    let client: OpenAIClient = OpenAIClient::new().unwrap();
    let embeddings: Vec<(String, Vec<f32>)> = client.generate_embeddings(chunks).unwrap();
    println!("Embeddings: {:?}", embeddings);

    // This will happen automatically in the future
    store.create_table().unwrap();

    // Insert the embeddings into the store this will be a better interface in the future
    for item in embeddings {
        store.store(item).unwrap();
    }
}
