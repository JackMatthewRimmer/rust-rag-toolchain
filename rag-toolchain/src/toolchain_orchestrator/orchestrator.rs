use std::io::Error;
use crate::toolchain_chunking::chunker::*;
use threadpool::ThreadPool;
use std::sync::mpsc::channel;

/// # generate_embeddings
/// raw_text: Iterator containing all the raw text values to create embeddings for
/// write_function: function with the side effect of writing the embeddings to an external source (e.g. pgvector) 
pub fn generate_embeddings(raw_text: impl ExactSizeIterator<Item = String>, write_function: fn(String) -> Result<(), Error>) {

  let no_tasks : usize = raw_text.len();
  let no_threads : usize = if no_tasks >= 10 { no_tasks/10 } else { 1 };
  let thread_pool : ThreadPool = ThreadPool::new(no_threads);
  let (sender, receiver) = channel::<Vec<String>>();

  // Spawn a thread for each piece of raw_text.
  for text in raw_text {
    let sender_clone = sender.clone();
    thread_pool.execute(move || {
      sender_clone.send(generate_chunks(&text, 500, 8000).unwrap()).unwrap();
    })
  }

  // Collect the chunks from the threads.
  let all_chunks: Vec<Vec<String>> = receiver.iter().take(no_tasks).collect();
}