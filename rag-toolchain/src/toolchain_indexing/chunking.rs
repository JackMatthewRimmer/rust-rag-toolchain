use tiktoken_rs::{cl100k_base, CoreBPE};

const OPEN_AI_MAX_EMBEDDING_TOKENS: usize = 8192;
/*
  IMPORTANT NOTE:
  Will Currently only work with the cl100k_base model
  as will be using text-embedding-ada-002
*/

/// # ChunkingError
/// Custom error type representing errors that can occur during chunking
#[derive(Debug, PartialEq, Eq)]
pub enum ChunkingError {
    WindowSizeTooLarge(String),
    TokenizationError(String),
    InvalidChunkSize(String),
}

/// # generate_chunks
/// function to generate chunks from raw text
pub fn generate_chunks(
    raw_text: &str,
    window_size: usize,
    chunk_tokens: usize,
) -> Result<Vec<String>, ChunkingError> {
    // validate arguments
    match chunk_tokens {
        ct if ct > OPEN_AI_MAX_EMBEDDING_TOKENS => {
            return Err(ChunkingError::InvalidChunkSize(format!(
                "Chunk size must be smaller than {}",
                OPEN_AI_MAX_EMBEDDING_TOKENS
            )))
        }
        ct if window_size >= ct => {
            return Err(ChunkingError::WindowSizeTooLarge(
                "Window size must be smaller than chunk size".to_string(),
            ))
        }
        _ => (),
    }

    // Generate token array from raw text
    let bpe: CoreBPE = cl100k_base().unwrap();
    let tokens: Vec<String> = get_tokens_iter(raw_text, &bpe)?;

    let mut chunks = Vec::new();
    let mut i = 0;
    while i < tokens.len() {
        let end = std::cmp::min(i + chunk_tokens, tokens.len());
        let chunk: String = tokens[i..end].to_vec().join("").trim().to_string();
        chunks.push(chunk);
        i += chunk_tokens - window_size;
    }
    return Ok(chunks);
}

/// # get_tokens_iter
/// Helper function to generate a token array from raw text
fn get_tokens_iter(text: &str, bpe: &CoreBPE) -> Result<Vec<String>, ChunkingError> {
    let tokens = bpe.split_by_token_iter(text, true);
    let mut unwrapped_tokens: Vec<String> = Vec::new();
    // Currently fail hard on tokenization failure
    for token in tokens {
        match token {
            Ok(token) => {
                unwrapped_tokens.push(token);
            }
            Err(error) => {
                return Err(ChunkingError::TokenizationError(error.to_string()));
            }
        }
    }
    return Ok(unwrapped_tokens);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_chunks_with_valid_input() {
        let raw_text: &str = "This is a test string";
        let window_size: usize = 1;
        let chunk_size: usize = 2;
        let chunks: Vec<String> =
            generate_chunks(raw_text, window_size, chunk_size).expect("Failed to generate chunks");
        println!("{:?}", chunks);
        assert_eq!(chunks.len(), 5);
        assert_eq!(
            chunks,
            vec!["This is", "is a", "a test", "test string", "string"]
        );
    }

    #[test]
    fn test_generate_chunks_with_empty_string() {
        let raw_text: &str = "";
        let window_size: usize = 1;
        let chunk_size: usize = 2;
        let chunks: Vec<String> =
            generate_chunks(raw_text, window_size, chunk_size).expect("Failed to generate chunks");
        println!("{:?}", chunks);
        assert_eq!(chunks.len(), 0);
        assert_eq!(chunks, Vec::<String>::new());
    }

    #[test]
    fn test_generate_chunks_with_invalid_arguments() {
        let raw_text: &str = "This is a test string";
        let window_size: usize = 2;
        let chunk_size: usize = 1;
        let chunks: ChunkingError = generate_chunks(raw_text, window_size, chunk_size)
            .expect_err("Failed to generate chunks");
        println!("{:?}", chunks);
        assert_eq!(
            chunks,
            ChunkingError::WindowSizeTooLarge(
                "Window size must be smaller than chunk size".to_string()
            )
        );
    }
}
