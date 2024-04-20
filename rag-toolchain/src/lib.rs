/// # Chains
///
/// The chains module represents a set of abstractions for common GenAI workflows
/// (which would have been a better name !). Examples of this can be RAG or interacting
/// with an LLM where the chat history is saved and resent with each request. This module
/// is essentially the core of what the library is trying to provide and the rest of the modules
/// are just tools in order to support these abstracted workflows.
pub mod chains;

/// # Chunkers
///
/// This module is aimed and the process of "chunking" or also referred to as "text splitting".
/// Wondering what that is ? put shortly its a set of methods in order to break down text into
/// smaller chunks / pieces. This is done to either keep embedded text short and consise or also
/// to ensure that what your trying to embed isnt larger than the token limit the embedding mode
/// has set out.
pub mod chunkers;

/// # Clients
///
/// The clients module is simply a place for all our implemented clients and there supporting code.
/// This is going to be common for any embedding model's and LLM's we support and the easiest is the
/// example of our OpenAI client.
pub mod clients;

/// # Common
///
/// The common module is a home for all our code that is shared across multiple modules. A prime example
/// of this would be any domain specific types that can appear across the library such as the [`common::Chunk`] type.
pub mod common;

/// # Loaders
///
/// The aim of this module is to provide some easy data integrations for you AI workflows. This could be as simple as
/// loading PDF's to potentially some third party services like notion. This has been less of a priority of the library
/// so far but hopefully in the future we can start to build this aspect out.
pub mod loaders;

/// # Retrievers
///
/// Now what are retrievers ? retrievers are a bridge between an embedding model and a vector database allowing you to query
/// with text and get text back without having to do the embedding of the query text manually.
pub mod retrievers;

/// # Stores
///
/// Stores are essentially just a reprisentation of a vector database allowing you to store any text into the vector database.
/// Like retrievers
pub mod stores;
