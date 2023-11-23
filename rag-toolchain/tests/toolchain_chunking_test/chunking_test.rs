use rag_toolchain::toolchain_indexing::chunking::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_chunks_with_valid_input() {
        let raw_text: &str = "This is a test string";
        let window_size: usize = 1;
        let chunk_size: usize = 2;
        let chunks: Vec<String> =
            generate_chunks(raw_text, window_size, chunk_size).expect("Failed to generate chunks");
        println!("{:?}", chunks);
        assert_eq!(chunks.len(), 5);
        assert_eq!(
            chunks,
            vec!["This is", "is a", "a test", "test string", "string"]
        );
    }

    #[test]
    fn test_generate_chunks_with_empty_string() {
        let raw_text: &str = "";
        let window_size: usize = 1;
        let chunk_size: usize = 2;
        let chunks: Vec<String> =
            generate_chunks(raw_text, window_size, chunk_size).expect("Failed to generate chunks");
        println!("{:?}", chunks);
        assert_eq!(chunks.len(), 0);
        assert_eq!(chunks, Vec::<String>::new());
    }

    #[test]
    fn test_generate_chunks_with_invalid_arguments() {
        let raw_text: &str = "This is a test string";
        let window_size: usize = 2;
        let chunk_size: usize = 1;
        let chunks: ChunkingError = generate_chunks(raw_text, window_size, chunk_size)
            .expect_err("Failed to generate chunks");
        println!("{:?}", chunks);
        assert_eq!(
            chunks,
            ChunkingError::WindowSizeTooLarge(
                "Window size must be smaller than chunk size".to_string()
            )
        );
    }
}
