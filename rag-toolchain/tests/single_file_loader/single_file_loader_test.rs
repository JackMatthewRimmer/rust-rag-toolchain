pub mod tests {
    use futures::StreamExt;
    use rag_toolchain::loaders::{LoadSource, SingleFileSource, SingleFileStream};

    #[test]
    fn test_when_file_exists() {
        let sut: SingleFileSource = SingleFileSource::new("tests/single_file_loader/test.txt");
        let result = sut.load().unwrap();
        assert_eq!(result, vec!["Hello World !\n"]);
    }

    #[test]
    fn test_when_file_doesnt_exist() {
        let sut: SingleFileSource = SingleFileSource::new("fake_file.txt");
        let err: std::io::Error = sut.load().unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound)
    }

    #[tokio::test]
    async fn test_stream() {
        let file = tokio::fs::File::open("tests/single_file_loader/test.txt")
            .await
            .unwrap();
        let mut stream = SingleFileStream::new(file);
        let expected_string: &[u8] = b"Hello World !\n";
        let mut index = 0;

        println!("{:?}", expected_string);

        while let Some(result) = stream.next().await {
            match result {
                Ok(byte) => {
                    println!("expected {:?}", expected_string[index]);
                    assert_eq!(byte, expected_string[index]);
                    index += 1;
                }
                Err(error) => panic!("Error: {}", error),
            }
        }
    }
}
