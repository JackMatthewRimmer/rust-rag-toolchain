pub mod tests {
    use rag_toolchain::loaders::{LoadSource, SingleFileSource};

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
}
