use std::{io::Error, thread::JoinHandle};
use crate::toolchain_chunking::chunker::*;
use std::thread;
use threadpool::ThreadPool;
use std::sync::mpsc::{channel, Sender};

/// # generate_embeddings
/// raw_text: Iterator containing all the raw text values to create embeddings for
/// write_function: function with the side effect of writing the embeddings to an external source (e.g. pgvector) 
pub fn generate_embeddings(raw_text: impl ExactSizeIterator<Item = String>, write_function: fn(String) -> Result<(), Error>) {

  let no_tasks = raw_text.len();
  let thread_pool = ThreadPool::new(50);
  let (sender, receiver) = channel::<Vec<String>>();

  // Spawn a thread for each piece of raw_text.
  for text in raw_text {
    let sender_clone = sender.clone();
    thread_pool.execute(move || {
      sender_clone.send(generate_chunks(&text, 1, 2).unwrap()).unwrap();
    })
  }

  // Collect the chunks from the threads.
  let all_chunks: Vec<Vec<String>> = receiver.iter().take(no_tasks).collect();
  println!("All chunks: {:?}", all_chunks.len());
  println!("All chunks: {:?}", all_chunks);
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
    let _embeddings = generate_embeddings(raw_text.into_iter(), write_function);
  }

}