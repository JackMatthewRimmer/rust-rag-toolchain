use crate::toolchain_orchestrator::traits::EmbeddingDataSource;
use std::fs::read_to_string;

/// # SingleFileSource
/// Reads a single file and returns the contents as a vector with one file
/// which is the file contents
pub struct SingleFileSource {
    path: String,
}

impl SingleFileSource {
    pub fn new(path: String) -> SingleFileSource {
        SingleFileSource { path }
    }
}

impl EmbeddingDataSource for SingleFileSource {
    fn read_source_data(&self) -> Result<Vec<String>, std::io::Error> {
        let file_contents: String = read_to_string(&self.path)?;
        return Ok(vec![file_contents]);
    }
}
