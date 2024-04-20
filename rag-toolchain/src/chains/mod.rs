/// # Chains
/// This chains represent common workflows that have been abstracted for
/// ease of use. Once all the required elements have been created you can
/// create a chain and call on it and it will execute the worklow under the
/// hood for you.
mod basic_rag_chain;
mod chat_history_chain;
mod types;
mod utils;

pub use basic_rag_chain::{
    BasicRAGChain, BasicRAGChainBuilder, BasicStreamedRAGChain, BasicStreamedRAGChainBuilder,
};
pub use chat_history_chain::ChatHistoryChain;
pub use types::{ChainError, RagChainError};
