use rag_toolchain::{
    chains::BasicRAGChain,
    clients::{
        OpenAIChatCompletionClient, OpenAIEmbeddingClient, OpenAIModel::Gpt3Point5, PromptMessage,
    },
    common::OpenAIEmbeddingModel::TextEmbeddingAda002,
    retrievers::{PostgresVectorRetriever, DistanceFunction},
    stores::PostgresVectorStore,
};

use std::num::NonZeroU32;

const SYSTEM_MESSAGE: &'static str =
    "You are to give straight forward answers using the supporting information you are provided";

#[tokio::main]
async fn main() {
    // In this example we are assuming there is already a pre populated vector store
    std::env::set_var("POSTGRES_USER", "postgres");
    std::env::set_var("POSTGRES_PASSWORD", "postgres");
    std::env::set_var("POSTGRES_HOST", "localhost");
    std::env::set_var("POSTGRES_DATABASE", "postgres");

    let store: PostgresVectorStore =
        PostgresVectorStore::try_new("embeddings", TextEmbeddingAda002)
            .await
            .unwrap();

    let embedding_client: OpenAIEmbeddingClient =
        OpenAIEmbeddingClient::try_new(TextEmbeddingAda002).unwrap();

    let retriever: PostgresVectorRetriever<OpenAIEmbeddingClient> =
        store.as_retriever(embedding_client, DistanceFunction::Cosine);

    let chat_client: OpenAIChatCompletionClient =
        OpenAIChatCompletionClient::try_new(Gpt3Point5).unwrap();

    let system_prompt: PromptMessage = PromptMessage::SystemMessage(SYSTEM_MESSAGE.into());

    let chain: BasicRAGChain<OpenAIChatCompletionClient, PostgresVectorRetriever<_>> =
        BasicRAGChain::builder()
            .system_prompt(system_prompt)
            .chat_client(chat_client)
            .retriever(retriever)
            .build();
    let user_message: PromptMessage =
        PromptMessage::HumanMessage("what kind of alcohol does Morwenna drink".into());
    let response = chain
        .invoke_chain(user_message, NonZeroU32::new(2).unwrap())
        .await
        .unwrap();

    println!("{}", response.content());
}
