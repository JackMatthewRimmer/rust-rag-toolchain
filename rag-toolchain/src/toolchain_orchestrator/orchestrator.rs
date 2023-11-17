use std::io::Error;
use crate::toolchain_chunking::chunker::*;
use std::thread;

/// # generate_embeddings
/// raw_text: Iterator containing all the raw text values to create embeddings for
/// write_function: function with the side effect of writing the embeddings to an external source (e.g. pgvector) 
pub fn generate_embeddings(raw_text: impl Iterator<Item = String>, write_function: fn(String) -> Result<(), Error>) {
  let mut handles = Vec::new();
  for text in raw_text {
    let handle = thread::spawn(move || {
      let chunks: Vec<String> = generate_chunks(&text, 1, 2).expect("Failed to generate chunks");
      println!("{:?}", chunks);
    });
    handles.push(handle);
  }

  for handle in handles {
    handle.join().unwrap();
  }
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
}