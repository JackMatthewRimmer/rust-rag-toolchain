use crate::loaders::traits::LoadSource;
use futures::Stream;
use std::pin::Pin;
use tokio::io::AsyncRead;
use tokio::io::ReadBuf;

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
        let file_contents: String = std::fs::read_to_string(&self.path)?;
        Ok(vec![file_contents])
    }
}

pub struct SingleFileStream {
    file: tokio::fs::File,
}

impl SingleFileStream {
    pub fn new(file: tokio::fs::File) -> SingleFileStream {
        SingleFileStream { file }
    }
}

impl Stream for SingleFileStream {
    type Item = std::io::Result<u8>;

    fn poll_next(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context,
    ) -> std::task::Poll<Option<Self::Item>> {
        let this = self.get_mut();
        let mut buffer = [0; 1];
        let mut read_buf = ReadBuf::new(&mut buffer);
        let file_pin = Pin::new(&mut this.file);

        match file_pin.poll_read(cx, &mut read_buf) {
            std::task::Poll::Ready(Ok(_)) => {
                let byte = read_buf.filled().get(0);
                return match byte {
                    Some(byte) => std::task::Poll::Ready(Some(Ok(*byte))),
                    None => std::task::Poll::Ready(None),
                };
            }
            std::task::Poll::Ready(Err(error)) => std::task::Poll::Ready(Some(Err(error))),
            std::task::Poll::Pending => std::task::Poll::Pending,
        }
    }
}
