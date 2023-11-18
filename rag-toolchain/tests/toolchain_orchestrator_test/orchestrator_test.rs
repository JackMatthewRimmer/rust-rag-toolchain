use rag_toolchain::toolchain_orchestrator::orchestrator::*;

#[cfg(test)]
mod tests {
    // Might be able to fully test this with a mock OpenAI client
    use super::*;

    #[test]
    fn test_builder_with_valid_inputs_builds_orchestrator() {
        let _orchestrator = Orchestrator::builder()
            .read_function(|| Ok(vec!["test".to_string()]))
            .write_function(|_text| Ok(()))
            .chunk_size(2)
            .window_size(1)
            .build();
    }

    #[test]
    fn test_builder_with_invalid_window_size_panics() {
        let result = std::panic::catch_unwind(|| {
            let _orchestrator = Orchestrator::builder()
                .read_function(|| Ok(vec!["test".to_string()]))
                .write_function(|_text| Ok(()))
                .chunk_size(2)
                .window_size(3)
                .build();
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_with_zero_window_size_panics() {
        let result = std::panic::catch_unwind(|| {
            let _orchestrator = Orchestrator::builder()
                .read_function(|| Ok(vec!["test".to_string()]))
                .write_function(|_text| Ok(()))
                .chunk_size(4)
                .window_size(0)
                .build();
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_with_zero_chunk_size_panics() {
        let result = std::panic::catch_unwind(|| {
            let _orchestrator = Orchestrator::builder()
                .read_function(|| Ok(vec!["test".to_string()]))
                .write_function(|_text| Ok(()))
                .chunk_size(0)
                .window_size(1)
                .build();
        });
        assert!(result.is_err());
    }
}
