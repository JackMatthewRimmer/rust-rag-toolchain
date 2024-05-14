# Rust RAG Toolchain

## Overview

`rust-rag-toolchain` is a Rust native library designed to empower developers with seamless access to common Retrieval Augmented Generation (RAG) workflows. It provides functionalities for embedding generation, storage, and retrieval, allowing developers to build AI applications with their own data.

## Features

- **Retrieval Augmented Generation (RAG):** Enhance your AI applications with better prompt completions based on your own knowledge base. See BasicRagChain ! 
- **PG Vector Support** Able to store and retrieve from a postgres database with pg_vector enabled !

- **OpenAI Support** Able to generate chat completions and embeddings via OpenAI to use an your AI workflows

- **Token Chunking** Able to chunk your text based on a token size.

## Getting Started

### Prerequisites

- Rust programming language installed. Visit [rust-lang.org](https://www.rust-lang.org/) for installation instructions. You will also need an Async Runtime; we recommend Tokio.

### Installation

Add the latest version of `rust-rag-toolchain` to your `Cargo.toml` file.

### Structure 

The library is structured into several modules, each responsible for a specific functionality:

- [`chains`](rag-toolchain/src/chains/): This module contains the implementation of the RAG workflows.

- [`chunkers`](rag-toolchain/src/chunkers/): This module provides utilities for breaking down data into manageable chunks.

- [`clients`](rag-toolchain/src/clients/): This module contains client implementations for interacting with various services.

- [`common`](rag-toolchain/src/common/): This module contains common utilities and helper functions used across the library.

- [`loaders`](rag-toolchain/src/loaders/): This module provides functionalities for loading and processing data.

- [`retrievers`](rag-toolchain/src/retrievers/): This module contains implementations for retrieving data from various sources.

- [`stores`](rag-toolchain/src/stores/): This module provides functionalities for storing and managing data.

### Code Examples

```rust

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
```


## Contributing

Contributions are welcome! If you have ideas for improvements or find any issues, please open an [issue](https://github.com/JackMatthewRimmer/rust-rag-toolchain/issues) or submit a pull request.

## Support

For any questions or assistance, feel free to contact the maintainers:

Please raise an issue and I will get back to you quickly

Enjoy leveraging the power of retrieval augmented generation and embedding generation with `rust-rag-toolchain`!
