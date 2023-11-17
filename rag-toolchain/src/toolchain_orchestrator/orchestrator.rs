use std::{io::Error, thread::JoinHandle};
use crate::toolchain_chunking::chunker::*;
use std::thread;

// Type alias for a thread handle in this context
type ChunkThreadHandle = JoinHandle<Vec<String>>;

/// # generate_embeddings
/// raw_text: Iterator containing all the raw text values to create embeddings for
/// write_function: function with the side effect of writing the embeddings to an external source (e.g. pgvector) 
pub fn generate_embeddings(raw_text: impl Iterator<Item = String>, write_function: fn(String) -> Result<(), Error>) {
  // Spawn a thread for each piece of raw_text.
  let handles: Vec<ChunkThreadHandle> = raw_text
    .map(|text| spawn_chunk_thread(text))
    .collect();

  // Collect the chunks from the threads.
  let all_chunks: Vec<Vec<String>> = handles.into_iter()
    .map(|handle: ChunkThreadHandle| handle.join().unwrap())
    .collect();
  println!("All chunks: {:?}", all_chunks);
}

fn spawn_chunk_thread(text: String) -> ChunkThreadHandle {
  return thread::spawn(move || {
    return generate_chunks(&text, 1, 2)
      .expect("Failed to generate chunks");
  });
}

pub fn generate_embeddings_no_threading(raw_text: impl Iterator<Item = String>, write_function: fn(String) -> Result<(), Error>) {

  let chunks: Vec<Vec<String>> = raw_text.map(|text| generate_chunks(&text, 1, 2).unwrap()).collect(); 
  println!("All chunks: {:?}", chunks);
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_generate_embeddings_with_valid_input() {
    let raw_text : Vec<String> = vec!["This is a test string".to_string(), "This is another test string".to_string()];
    let write_function : fn(String) -> Result<(), Error> = |x| Ok(());
    generate_embeddings(raw_text.into_iter(), write_function);
  }

  #[test]
  fn test_generate_embeddings_with_large_input() {
    let raw_text: Vec<String> = vec!["This is a test string".to_string(); 100];
    let write_function: fn(String) -> Result<(), Error> = |x| Ok(());
    generate_embeddings(raw_text.into_iter(), write_function);
  }

  #[test]
  fn test_generate_embeddings_no_threading_with_valid_input() {
    let raw_text: Vec<String> = vec!["This is a test string".to_string(); 100];
    let write_function : fn(String) -> Result<(), Error> = |x| Ok(());
    generate_embeddings_no_threading(raw_text.into_iter(), write_function);
  }
}