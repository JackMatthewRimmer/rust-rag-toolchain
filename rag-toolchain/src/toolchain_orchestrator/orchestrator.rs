use crate::toolchain_chunking::chunker::*;
use std::io::Error;
use std::sync::mpsc::channel;
use threadpool::ThreadPool;

pub enum OrchestratorError {
    ReadError(Error),
    WriteError(Error),
    ChunkingError(ChunkingError),
    EmbeddingError(String), // will likely be something from the OpenAI client,
}

/// # Orchestrator
/// executes the tasks of the toolchain
/// Should be built through a set of builder functions
/// which define source and destination specific read and write functions
pub struct Orchestrator {
    read_function: fn() -> Result<Vec<String>, Error>,
    write_function: fn(String) -> Result<(), Error>,
    chunk_size: usize,
    window_size: usize,
}

impl Orchestrator {
    pub fn execute(&self) -> Result<(), OrchestratorError> {
        let raw_text = match (self.read_function)() {
            Ok(text) => text,
            Err(error) => return Err(OrchestratorError::ReadError(error)),
        };
        let chunks: Vec<Vec<String>> =
            Orchestrator::execute_chunk_task(raw_text, self.chunk_size, self.window_size);

        let embeddings: Vec<(String, Vec<f32>)> = Orchestrator::execute_embeddings_task(chunks);
        match Orchestrator::execute_write_task(embeddings) {
            Ok(()) => return Ok(()),
            Err(error) => return Err(OrchestratorError::WriteError(error)),
        };
    }
    fn execute_chunk_task(
        raw_text: Vec<String>,
        chunk_size: usize,
        window_size: usize,
    ) -> Vec<Vec<String>> {
        let no_tasks: usize = raw_text.len();
        let no_threads: usize = if no_tasks >= 10 { no_tasks / 10 } else { 1 }; // potentially not most efficient way to do this
        let thread_pool: ThreadPool = ThreadPool::new(no_threads);
        let (sender, receiver) = channel::<Vec<String>>();

        // Spawn a thread for each piece of raw_text.
        for text in raw_text {
            let sender_clone = sender.clone();
            thread_pool.execute(move || {
                sender_clone
                    .send(generate_chunks(&text, window_size, chunk_size).unwrap())
                    .unwrap();
            })
        }

        // Collect the chunks from the threads.
        return receiver.iter().take(no_tasks).collect();
    }

    fn execute_embeddings_task(chunks: Vec<Vec<String>>) -> Vec<(String, Vec<f32>)> {
        // This is where we would send each chunk to openAI
        !unimplemented!()
    }

    fn execute_write_task(embeddings: Vec<(String, Vec<f32>)>) -> Result<(), Error> {
        // this is where we write the embeddings to the destination
        !unimplemented!()
    }
}
