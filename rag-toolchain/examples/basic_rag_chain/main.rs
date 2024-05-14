use rag_toolchain::{
    chains::BasicRAGChain,
    clients::{
        OpenAIChatCompletionClient, OpenAIEmbeddingClient, OpenAIModel::Gpt3Point5Turbo,
        PromptMessage,
    },
    common::OpenAIEmbeddingModel::TextEmbeddingAda002,
    retrievers::{DistanceFunction, PostgresVectorRetriever},
    stores::PostgresVectorStore,
};

use std::num::NonZeroU32;

const SYSTEM_MESSAGE: &'static str =
    "You are to give straight forward answers using the supporting information you are provided";

#[tokio::main]
async fn main() {
    // Initialize the PostgresVectorStore
    let store: PostgresVectorStore =
        PostgresVectorStore::try_new("embeddings", TextEmbeddingAda002)
            .await
            .unwrap();

    // Create a new embedding client
    let embedding_client: OpenAIEmbeddingClient =
        OpenAIEmbeddingClient::try_new(TextEmbeddingAda002).unwrap();

    // Convert our store into a retriever
    let retriever: PostgresVectorRetriever<OpenAIEmbeddingClient> =
        store.as_retriever(embedding_client, DistanceFunction::Cosine);

    // Create a new chat client
    let chat_client: OpenAIChatCompletionClient =
        OpenAIChatCompletionClient::try_new(Gpt3Point5Turbo).unwrap();

    // Define our system prompt
    let system_prompt: PromptMessage = PromptMessage::SystemMessage(SYSTEM_MESSAGE.into());

    // Create a new BasicRAGChain with over our open ai chat client and postgres vector retriever
    let chain: BasicRAGChain<OpenAIChatCompletionClient, PostgresVectorRetriever<_>> =
        BasicRAGChain::builder()
            .system_prompt(system_prompt)
            .chat_client(chat_client)
            .retriever(retriever)
            .build();
    // Define our user prompt
    let user_message: PromptMessage =
        PromptMessage::HumanMessage("what kind of alcohol does Morwenna drink".into());

    // Invoke the chain. Under the hood this will retrieve some similar text from
    // the retriever and then use the chat client to generate a response.
    let response = chain
        .invoke_chain(user_message, NonZeroU32::new(2).unwrap())
        .await
        .unwrap();

    println!("{}", response.content());
}
