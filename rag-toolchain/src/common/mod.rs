/// # Common
/// This module contains common types and traits used across the project
mod embedding_shared;
#[cfg(feature = "pg_vector")]
mod postgres_shared;
mod types;

pub use embedding_shared::*;
pub use types::*;
