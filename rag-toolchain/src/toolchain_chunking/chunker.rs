use tiktoken_rs::{CoreBPE, cl100k_base};

/* 
  IMPORTANT NOTE:
  Will Currently only work with the cl100k_base model 
  as will be using text-embedding-ada-002
*/

/// # ChunkingError
/// Custom error type representing errors that can occur during chunking
pub enum ChunkingError {
  WindowSizeTooLarge(String),
  TokenizationError(String),
}

/// # generate_chunks
/// function to generate chunks from raw text
pub fn generate_chunks(raw_text: &str, window_size: usize, chunk_size: usize) -> Result<Vec<String>, ChunkingError> {

  // validate arguments
  if window_size >= chunk_size {
    return Err(ChunkingError::WindowSizeTooLarge("Window size must be smaller than chunk size".to_string()))
  }

  // Generate token array from raw text
  let bpe: CoreBPE = cl100k_base().unwrap();
  let tokens: Vec<String> = get_tokens_iter(raw_text, &bpe)?;

  let mut chunks = Vec::new();
  let mut i = 0;
  while i < tokens.len() {
      let end = std::cmp::min(i + chunk_size, tokens.len());
      let chunk: String = tokens[i..end]
          .to_vec()
          .join("")
          .trim()
          .to_string();
      chunks.push(chunk);
      i += chunk_size - window_size;
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
      Ok(token) => {unwrapped_tokens.push(token);},
      Err(error) => {return Err(ChunkingError::TokenizationError(error.to_string()));}
    }
  }
  return Ok(unwrapped_tokens);
}
