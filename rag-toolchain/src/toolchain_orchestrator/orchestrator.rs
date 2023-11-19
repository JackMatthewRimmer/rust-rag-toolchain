use crate::toolchain_chunking::chunker::*;
use crate::toolchain_embeddings::client::OpenAIClient;
use crate::toolchain_orchestrator::external::*;
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
pub struct Orchestrator {
    source: Box<dyn Source>,
    destination: Box<dyn Destination>,
    openai_client: OpenAIClient,
    chunk_size: usize,
    window_size: usize,
}

impl Orchestrator {
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
    pub fn builder() -> BaseOrchestratorBuilder {
        BaseOrchestratorBuilder::default()
    }

    /// # Examples
    /// ```
    /// Orchestrator.execute().expect("Orchestration failed")
    /// ```
    /// # Returns
    /// A result containing either ```Ok(())``` or an error of type [`OrchestratorError`]
    pub fn execute(&self) -> Result<(), OrchestratorError> {
        let raw_text = match self.source.read_from_source() {
            Ok(text) => text,
            Err(error) => return Err(OrchestratorError::ReadError(error)),
        };

        let chunks: Vec<Vec<String>> =
            Orchestrator::execute_chunk_task(raw_text, self.chunk_size, self.window_size);

        let embeddings: Vec<(String, Vec<f32>)> =
            match Orchestrator::execute_embeddings_task(chunks) {
                Ok(embeddings) => embeddings,
                Err(error) => return Err(OrchestratorError::EmbeddingError(error.to_string())),
            };

        match Orchestrator::execute_write_task(embeddings) {
            Ok(()) => return Ok(()),
            Err(error) => return Err(OrchestratorError::WriteError(error)),
        };
    }

    // Lacking error handling here
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
        source: Box<dyn Source>,
        destination: Box<dyn Destination>,
        openai_client: OpenAIClient,
        chunk_size: usize,
        window_size: usize,
    ) -> Orchestrator {
        // Constructor constraints
        assert!(chunk_size > 0);
        assert!(window_size > 0);
        assert!(chunk_size > window_size);
        return Orchestrator {
            source,
            destination,
            openai_client,
            chunk_size,
            window_size,
        };
    }
}

/// # BaseOrchestratorBuilder
/// This is the base for all source and destination specific orchestrators.
/// These extensions should be built on top of this base builder to preserve field constraints
pub struct BaseOrchestratorBuilder {
    source: Box<dyn Source>,
    destination: Box<dyn Destination>,
    openai_client: OpenAIClient,
    chunk_size: usize,
    window_size: usize,
}

// Builder functions
impl BaseOrchestratorBuilder {
    pub fn read_function(mut self, source: Box<dyn Source>) -> Self {
        self.source = source;
        return self;
    }

    pub fn write_function(mut self, destination: Box<dyn Destination>) -> Self {
        self.destination = destination;
        return self;
    }

    pub fn openai_client(mut self, openai_client: OpenAIClient) -> Self {
        self.openai_client = openai_client;
        return self;
    }

    pub fn chunk_size(mut self, chunk_size: usize) -> Self {
        assert!(chunk_size > 0);
        self.chunk_size = chunk_size;
        return self;
    }

    pub fn window_size(mut self, window_size: usize) -> Self {
        assert!(window_size > 0);
        self.window_size = window_size;
        return self;
    }

    pub fn build(self) -> Orchestrator {
        return Orchestrator::new(
            self.source,
            self.destination,
            self.openai_client,
            self.chunk_size,
            self.window_size,
        );
    }
}

// Default values for the builder
impl Default for BaseOrchestratorBuilder {
    fn default() -> Self {
        return BaseOrchestratorBuilder {
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

impl Source for DefaultSource {
    /// # Errors
    ///
    /// Returns an error of type `Error` as the source was not set.
    fn read_from_source(&self) -> Result<Vec<String>, Error> {
        return Err(Error::new(ErrorKind::NotFound, "source not set"));
    }
}

/// Represents a default destination.
struct DefaultDestination;

impl Destination for DefaultDestination {
    /// # Errors
    ///
    /// Returns an error of type `Error` as destination was not set.
    fn write_to_dest(&self, _: String) -> Result<(), Error> {
        return Err(Error::new(ErrorKind::NotFound, "destination not set"));
    }
}
