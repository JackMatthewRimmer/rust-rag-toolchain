use crate::toolchain_indexing::chunking::*;
use crate::toolchain_indexing::traits::*;
use std::io::Error;
use std::sync::mpsc::channel;
use threadpool::ThreadPool;
use typed_builder::TypedBuilder;

/// # EmbeddingTaskError
/// Errors that can occur during the task execution
pub enum EmbeddingTaskError {
    /// Error executing the read function
    ReadError(Error),
    /// Error executing the write function
    WriteError(Error),
    /// Error executing [`generate_chunks`]
    ChunkingError(ChunkingError),
    /// Error generating the embeddings from OpenAI
    EmbeddingError(String), // will likely be something from the OpenAI client,
}

/// # EmbeddingTaskArgumentError
/// Errors that Occur when bad arguments are passed when constructing the task
pub enum EmbeddingTaskArgumentError {
    /// Error when the chunk size is 0 as we cannot embed 0 tokens
    ChunkSizeIsZero(String),
    WindowSizeIsLargerThanChunkSize(String),
}

/// # GenerateEmbeddingTask
///  Represents a task that reads from a source, chunks the text,
///  generates embeddings, and writes the embeddings to a destination.
///
/// # Builder Argument Constraints
///
/// 1. `chunk_size` must be greater than 0
/// 2. `window_size` must be less than or equal to `chunk_size`
///
/// /// # Fields
/// * `source`: The source to read from.
/// * `destination`: The destination to write the embeddings to.
/// * `embedding_client`: The client used to generate the embeddings.
/// * `chunk_size`: The size of the chunks to generate.
/// * `chunk_overlap`: The size of the overlap between chunks.
///
/// # Errors
/// [`EmbeddingTaskError`] is returned if any of the above steps fail
/// [`EmbeddingTaskArgumentError`] is returned if the arguments passed to the builder are invalid
#[derive(TypedBuilder)]
// set generated build method to private an under a different name so we can implement one with our own logic
#[builder(build_method(vis="", name=__build))]
pub struct GenerateEmbeddingTask {
    source: Box<dyn LoadSource>,
    destination: Box<dyn EmbeddingStore>,
    embedding_client: Box<dyn EmbeddingClient>,
    chunk_size: usize,
    chunk_overlap: usize,
}

impl GenerateEmbeddingTask {
    /// # Execute
    /// Carries out the following steps:
    /// 1. Reads the raw text from the source
    /// 2. Generates chunks from the raw text
    /// 3. Generates embeddings from the chunks
    /// 4. Writes the embeddings to the destination
    ///
    /// # Examples
    /// ```
    /// Orchestrator.execute().expect("Orchestration failed")
    /// ```
    /// # Returns
    /// A result containing either ```Ok(())``` or an error of type [`EmbeddingTaskError`]
    pub fn execute(&self) -> Result<(), EmbeddingTaskError> {
        let raw_text = match self.source.read_source_data() {
            Ok(text) => text,
            Err(error) => return Err(EmbeddingTaskError::ReadError(error)),
        };

        let chunks: Vec<Vec<String>> =
            GenerateEmbeddingTask::chunk(raw_text, self.chunk_size, self.chunk_overlap);

        let embeddings: Vec<(String, Vec<f32>)> = match GenerateEmbeddingTask::embed(chunks) {
            Ok(embeddings) => embeddings,
            Err(error) => return Err(EmbeddingTaskError::EmbeddingError(error.to_string())),
        };

        match GenerateEmbeddingTask::store(embeddings) {
            Ok(()) => return Ok(()),
            Err(error) => return Err(EmbeddingTaskError::WriteError(error)),
        };
    }

    /// # Chunk
    /// Generates chunks from the raw text in parallel
    ///
    /// # Arguments
    /// * `raw_text`: The raw text loaded from the [`LoadSource`]
    /// * `chunk_size`: The size of the chunks to generate
    /// * `window_size`: The size of the overlap between chunks
    ///
    /// # Returns
    /// A vector of chunks for each piece of input text
    fn chunk(raw_text: Vec<String>, chunk_size: usize, window_size: usize) -> Vec<Vec<String>> {
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

    // Error returned here will change to something from the OpenAI client
    fn embed(chunks: Vec<Vec<String>>) -> Result<Vec<(String, Vec<f32>)>, Error> {
        // This is where we would send each chunk to openAI to get embeddings
        !unimplemented!()
    }

    /// # Store
    /// Writes the embeddings to the ['EmbeddingStore']
    ///
    /// # Arguments
    /// * `embeddings`: The embeddings to write to the destination along side the original chunk text
    ///
    /// # Returns
    /// [`Ok(Vec<(String, Vec<f32>))`] if the embeddings were successfully written to the destination
    /// [`Err(Error)`] if there was an error writing the embeddings to the destination
    fn store(embeddings: Vec<(String, Vec<f32>)>) -> Result<Vec<(String, Vec<f32>)>, Error> {
        // this is where we write the embeddings to the destination
        !unimplemented!()
    }
}

/// # GenerateEmbeddingTaskBuilder
/// This is mostly generated by the [`TypedBuilder`] macro, However
/// we need to implement our own build method so we can implement our own logic
/// to check the arguments passed to the builder and return an error if they are invalid
impl
    GenerateEmbeddingTaskBuilder<(
        (Box<dyn LoadSource>,),
        (Box<dyn EmbeddingStore>,),
        (Box<dyn EmbeddingClient>,),
        (usize,),
        (usize,),
    )>
{
    /// # Returns
    /// `Result<GenerateEmbeddingTask, EmbeddingTaskArgumentError>` - Will return an error if the arguments passed to the builder
    /// are invalid. See ['EmbeddingTaskArgumentError']
    pub fn build(self) -> Result<GenerateEmbeddingTask, EmbeddingTaskArgumentError> {
        let (source, destination, embedding_client, chunk_size, chunk_overlap) = self.fields;
        let source = source.0;
        let destination = destination.0;
        let embedding_client = embedding_client.0;
        let chunk_size = chunk_size.0;
        let chunk_overlap = chunk_overlap.0;

        // Constructor constraints
        if chunk_size == 0 {
            return Err(EmbeddingTaskArgumentError::ChunkSizeIsZero(
                "Chunk size must be greater than 0".to_string(),
            ));
        }

        if chunk_overlap > chunk_size {
            return Err(EmbeddingTaskArgumentError::WindowSizeIsLargerThanChunkSize(
                "Window size must be less than or equal to chunk size".to_string(),
            ));
        }

        Ok(GenerateEmbeddingTask {
            source,
            destination,
            embedding_client,
            chunk_size,
            chunk_overlap,
        })
    }
}
