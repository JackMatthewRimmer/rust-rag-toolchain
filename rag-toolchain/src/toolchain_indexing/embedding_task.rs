use crate::toolchain_embeddings::openai_embeddings::OpenAIEmbeddingClient;
use crate::toolchain_indexing::chunking::*;
use crate::toolchain_indexing::traits::*;
use std::io::Error;
use std::sync::mpsc::channel;
use threadpool::ThreadPool;
use typed_builder::TypedBuilder;

/* Notes:
This task is probably impossible to make generic or at least should be bound to openAI embeddings
*/

/// # EmbeddingTaskError
/// Errors that can occur during the task execution
#[derive(Debug)]
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
#[derive(Debug)]
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
/// # Fields
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
    embedding_client: Box<dyn OpenAIEmbeddingClient>,
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
    /// use crate::rag_toolchain::toolchain_indexing::destinations::PgVector;
    /// use crate::rag_toolchain::toolchain_indexing::sources::SingleFileSource;
    /// use crate::rag_toolchain::toolchain_indexing::embedding_task::GenerateEmbeddingTask;
    /// use crate::rag_toolchain::toolchain_embeddings::openai_embeddings::OpenAIClient;
    ///
    /// // This should be set in a .env file in the root of the project
    /// std::env::set_var("POSTGRES_USERNAME", "postgres");
    /// std::env::set_var("POSTGRES_PASSWORD", "password");
    /// std::env::set_var("POSTGRES_HOST", "localhost");
    /// std::env::set_var("POSTGRES_DATABASE", "pg_vector");
    ///
    /// let source = SingleFileSource::new("path"); // create a source
    /// let destination = PgVector::new("table_name").unwrap(); // create a destination
    /// let embedding_client = OpenAIClient::new(); // create an embedding client
    /// let chunk_size = 8192; // specify a chunk size
    /// let chunk_overlap = 1000; // specify a chunk overlap
    ///
    /// let task = GenerateEmbeddingTask::builder()
    ///     .source(Box::new(source))
    ///     .destination(Box::new(destination))
    ///     .embedding_client(Box::new(embedding_client))
    ///     .chunk_size(chunk_size)
    ///     .chunk_overlap(chunk_overlap)
    ///     .build()
    ///     .unwrap();
    ///
    /// // Uncomment this when its implemented
    /// // task.execute()
    ///     //.expect("failed to generate and persist embeddings");
    /// ```
    /// # Returns
    /// A result containing either ```Ok(Vec<(String, Vec<f32>)>)``` or an error of type [`EmbeddingTaskError`]
    pub fn execute(&self) -> Result<Vec<(String, Vec<f32>)>, EmbeddingTaskError> {
        let raw_text = match self.source.load() {
            Ok(text) => text,
            Err(error) => return Err(EmbeddingTaskError::ReadError(error)),
        };

        let chunks: Vec<String> =
            GenerateEmbeddingTask::chunk(raw_text, self.chunk_size, self.chunk_overlap).concat();

        let embeddings: Vec<(String, Vec<f32>)> = match self.embed(chunks) {
            Ok(embeddings) => embeddings,
            Err(error) => return Err(EmbeddingTaskError::EmbeddingError(error.to_string())),
        };

        match self.store(embeddings) {
            Ok(embeddings) => return Ok(embeddings),
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
    fn embed(&self, chunks: Vec<String>) -> Result<Vec<(String, Vec<f32>)>, Error> {
        // This is where we would send each chunk to openAI to get embeddings

        let mut embeddings: Vec<(String, Vec<f32>)> = Vec::new();

        // This should leverage batches if possible
        for chunk in chunks {
            match self.embedding_client.generate_embeddings(vec![chunk.clone()]) {
                Ok(embedding) => embeddings.push((chunk, embedding)),
                Err(error) => return Err(error),
            };
        }

        return Ok(embeddings)
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
    fn store(&self, embeddings: Vec<(String, Vec<f32>)>) -> Result<Vec<(String, Vec<f32>)>, Error> {
        // Probably need some retry mechanism here maybe ?
        for embedding in &embeddings {
            self.destination.store(embedding.clone())?;
        }
        return Ok(embeddings);
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
        (Box<dyn OpenAIEmbeddingClient>,),
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

#[cfg(test)]
mod tests {

    struct TestHelper {}
    impl LoadSource for TestHelper {
        fn load(&self) -> Result<Vec<String>, Error> {
            Ok(vec!["test".to_string()])
        }
    }
    impl EmbeddingStore for TestHelper {
        fn store(&self, _text: (String, Vec<f32>)) -> Result<(), Error> {
            Ok(())
        }
    }
    impl OpenAIEmbeddingClient for TestHelper {
        fn generate_embeddings(&self, text: Vec<String>) -> Result<Vec<f32>, Error> {
            Ok(vec![0.0])
        }
    }

    // Might be able to fully test this with a mock OpenAI client
    use super::*;

    #[test]
    fn test_builder_with_valid_inputs_builds_orchestrator() {
        let test_source = Box::new(TestHelper {});
        let test_destination = Box::new(TestHelper {});
        let _orchestrator = GenerateEmbeddingTask::builder()
            .source(test_source)
            .destination(test_destination)
            .embedding_client(Box::new(TestHelper {}))
            .chunk_size(2)
            .chunk_overlap(1)
            .build();
    }
}
