use rag_toolchain::toolchain_orchestrator::embedding_client::*;
use rag_toolchain::toolchain_orchestrator::traits::*;
use std::io::Error;

#[cfg(test)]
mod tests {

    struct TestHelper {}
    impl EmbeddingDataSource for TestHelper {
        fn read_source_data(&self) -> Result<Vec<String>, Error> {
            Ok(vec!["test".to_string()])
        }
    }
    impl EmbeddingDestination for TestHelper {
        fn write_embedding(&self, _text: (String, Vec<f32>)) -> Result<(), Error> {
            Ok(())
        }
    }

    // Might be able to fully test this with a mock OpenAI client
    use super::*;

    #[test]
    fn test_builder_with_valid_inputs_builds_orchestrator() {
        let test_source = Box::new(TestHelper {});
        let test_destination = Box::new(TestHelper {});
        let _orchestrator = EmbeddingClient::builder()
            .source(test_source)
            .destination(test_destination)
            .chunk_size(2)
            .window_size(1)
            .build();
    }

    #[test]
    fn test_builder_with_invalid_window_size_panics() {
        let test_source = Box::new(TestHelper {});
        let test_destination = Box::new(TestHelper {});
        let result = std::panic::catch_unwind(|| {
            let _orchestrator = EmbeddingClient::builder()
                .source(test_source)
                .destination(test_destination)
                .chunk_size(2)
                .window_size(3)
                .build();
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_with_zero_window_size_panics() {
        let test_source = Box::new(TestHelper {});
        let test_destination = Box::new(TestHelper {});
        let result = std::panic::catch_unwind(|| {
            let _orchestrator = EmbeddingClient::builder()
                .source(test_source)
                .destination(test_destination)
                .chunk_size(4)
                .window_size(0)
                .build();
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_with_zero_chunk_size_panics() {
        let test_source = Box::new(TestHelper {});
        let test_destination = Box::new(TestHelper {});
        let result = std::panic::catch_unwind(|| {
            let _orchestrator = EmbeddingClient::builder()
                .source(test_source)
                .destination(test_destination)
                .chunk_size(0)
                .window_size(1)
                .build();
        });
        assert!(result.is_err());
    }
}
