use crate::loaders::traits::LoadSource;
use std::fs::read_to_string;

/// # [`SingleFileSource`]
/// Reads a single file and returns the contents as a vector with one file
/// which is the file contents
pub struct SingleFileSource {
    /// File path
    path: String,
}

impl SingleFileSource {
    pub fn new(path: impl Into<String>) -> SingleFileSource {
        SingleFileSource { path: path.into() }
    }
}

impl LoadSource for SingleFileSource {
    type ErrorType = std::io::Error;
    fn load(&self) -> Result<Vec<String>, Self::ErrorType> {
        let file_contents: String = read_to_string(&self.path)?;
        Ok(vec![file_contents])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_when_file_doesnt_exist() {
        let sut: SingleFileSource = SingleFileSource::new("fake_file.txt");
        let err: std::io::Error = sut.load().unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound)
    }
}
