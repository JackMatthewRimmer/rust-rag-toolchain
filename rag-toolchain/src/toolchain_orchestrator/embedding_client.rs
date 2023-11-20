use crate::toolchain_chunking::chunker::*;
use crate::toolchain_embeddings::client::OpenAIClient;
use crate::toolchain_orchestrator::traits::*;
use std::io::{Error, ErrorKind};
use std::sync::mpsc::channel;
use threadpool::ThreadPool;

/// # OrchestratorError
/// Errors that can occur during the orchestration process
pub enum OrchestratorError {
    /// Error executing the read function
    ReadError(Error),
    /// Error executing the write function
    WriteError(Error),
    /// Error executing [`generate_chunks`]
    ChunkingError(ChunkingError),
    /// Error generating the embeddings from OpenAI
    EmbeddingError(String), // will likely be something from the OpenAI client,
}

/// # Orchestrator
/// Executes each task of the toolchain which are...
///
/// 1. Read from the source
/// 2. Chunk the text
/// 3. Get embeddings from OpenAI
/// 4. Write to the destination
/// # functions
/// [`Orchestrator::builder`] returns a builder for the orchestrator
///
/// [`Orchestrator::execute`] executes the above steps
pub struct EmbeddingClient {
    source: Box<dyn EmbeddingDataSource>,
    destination: Box<dyn EmbeddingDestination>,
    openai_client: OpenAIClient,
    chunk_size: usize,
    window_size: usize,
}

impl EmbeddingClient {
    /// # Examples
    /// ```
    /// let orch: Orchestrator = Orchestrator::builder()
    ///     .read_function(read_function)
    ///     .write_function(write_function)
    ///     .openai_client(OpenAIClient::new())
    ///     .chunk_size(1024)
    ///     .window_size(128)
    ///     .build()
    /// ```
    /// # Returns
    /// An [`Orchestrator`] with the given parameters
    pub fn builder() -> EmbeddingClientBuilder {
        EmbeddingClientBuilder::default()
    }

    /// # Examples
    /// ```
    /// Orchestrator.execute().expect("Orchestration failed")
    /// ```
    /// # Returns
    /// A result containing either ```Ok(())``` or an error of type [`OrchestratorError`]
    pub fn execute(&self) -> Result<(), OrchestratorError> {
        let raw_text = match self.source.read_source_data() {
            Ok(text) => text,
            Err(error) => return Err(OrchestratorError::ReadError(error)),
        };

        let chunks: Vec<Vec<String>> =
            EmbeddingClient::execute_chunk_task(raw_text, self.chunk_size, self.window_size);

        let embeddings: Vec<(String, Vec<f32>)> =
            match EmbeddingClient::execute_embeddings_task(chunks) {
                Ok(embeddings) => embeddings,
                Err(error) => return Err(OrchestratorError::EmbeddingError(error.to_string())),
            };

        match EmbeddingClient::execute_write_task(embeddings) {
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

    // Error returned here will change to something from the OpenAI client
    fn execute_embeddings_task(chunks: Vec<Vec<String>>) -> Result<Vec<(String, Vec<f32>)>, Error> {
        // This is where we would send each chunk to openAI to get embeddings
        !unimplemented!()
    }

    fn execute_write_task(embeddings: Vec<(String, Vec<f32>)>) -> Result<(), Error> {
        // this is where we write the embeddings to the destination
        !unimplemented!()
    }

    fn new(
        source: Box<dyn EmbeddingDataSource>,
        destination: Box<dyn EmbeddingDestination>,
        openai_client: OpenAIClient,
        chunk_size: usize,
        window_size: usize,
    ) -> EmbeddingClient {
        // Constructor constraints
        assert!(chunk_size > 0);
        assert!(window_size > 0);
        assert!(chunk_size > window_size);
        return EmbeddingClient {
            source,
            destination,
            openai_client,
            chunk_size,
            window_size,
        };
    }
}

/// # BaseOrchestratorBuilder
/// Builder Struct used for creating an instance of [`Orchestrator`]
pub struct EmbeddingClientBuilder {
    source: Box<dyn EmbeddingDataSource>,
    destination: Box<dyn EmbeddingDestination>,
    openai_client: OpenAIClient,
    chunk_size: usize,
    window_size: usize,
}

impl EmbeddingClientBuilder {
    /// # Arguments
    /// * `source` - A struct with implements the [`Source`] trait
    pub fn source(mut self, source: Box<dyn EmbeddingDataSource>) -> Self {
        self.source = source;
        return self;
    }

    /// # Arguments
    /// * `destination` - A struct with implements the [`Destination`] trait
    pub fn destination(mut self, destination: Box<dyn EmbeddingDestination>) -> Self {
        self.destination = destination;
        return self;
    }

    pub fn openai_client(mut self, openai_client: OpenAIClient) -> Self {
        self.openai_client = openai_client;
        return self;
    }

    /// # Arguments
    /// * `chunk_size` - The size of each chunk in tokens which must be greater than zero
    /// and less than 8192
    ///
    /// # Panics
    /// Panics if `chunk_size` is greater than 8192
    /// Panics if `chunk_size` is less than 1
    pub fn chunk_size(mut self, chunk_size: usize) -> Self {
        assert!(chunk_size < 8192); // Max tokens for text-embedding-ada-002
        assert!(chunk_size > 0);
        self.chunk_size = chunk_size;
        return self;
    }

    /// # Arguments
    /// * `window_size` - The size of the overlap between each chunk in tokens which must be
    /// greater than zero and less than `chunk_size`
    ///
    /// # Panics
    /// Panics if `window_size` is less than 1
    pub fn window_size(mut self, window_size: usize) -> Self {
        assert!(window_size > 0);
        self.window_size = window_size;
        return self;
    }

    /// # Returns
    /// An instance of [`Orchestrator`] with the given parameters.
    /// All fields must be set
    pub fn build(self) -> EmbeddingClient {
        return EmbeddingClient::new(
            self.source,
            self.destination,
            self.openai_client,
            self.chunk_size,
            self.window_size,
        );
    }
}

// Default values for the builder
impl Default for EmbeddingClientBuilder {
    fn default() -> Self {
        return EmbeddingClientBuilder {
            source: Box::new(DefaultSource {}),
            destination: Box::new(DefaultDestination {}),
            openai_client: OpenAIClient::new(),
            chunk_size: 0,
            window_size: 0,
        };
    }
}

/// Represents a default source.
struct DefaultSource;

impl EmbeddingDataSource for DefaultSource {
    /// # Errors
    ///
    /// Returns an error of type `Error` as the source was not set.
    fn read_source_data(&self) -> Result<Vec<String>, Error> {
        return Err(Error::new(ErrorKind::NotFound, "source not set"));
    }
}

/// Represents a default destination.
struct DefaultDestination;

impl EmbeddingDestination for DefaultDestination {
    /// # Errors
    ///
    /// Returns an error of type `Error` as destination was not set.
    fn write_embedding(&self, _: (String, Vec<f32>)) -> Result<(), Error> {
        return Err(Error::new(ErrorKind::NotFound, "destination not set"));
    }
}
