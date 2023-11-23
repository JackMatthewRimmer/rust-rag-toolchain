use rag_toolchain::toolchain_indexing::embedding_task::*;
use rag_toolchain::toolchain_indexing::traits::*;
use std::io::Error;

#[cfg(test)]
mod tests {

    struct TestHelper {}
    impl LoadSource for TestHelper {
        fn read_source_data(&self) -> Result<Vec<String>, Error> {
            Ok(vec!["test".to_string()])
        }
    }
    impl EmbeddingStore for TestHelper {
        fn write_embedding(&self, _text: (String, Vec<f32>)) -> Result<(), Error> {
            Ok(())
        }
    }
    impl EmbeddingClient for TestHelper {
        fn generate_embeddings(&self) -> Result<Vec<f32>, Error> {
            Ok(vec![0.0])
        }
    }

    // Might be able to fully test this with a mock OpenAI client
    use super::*;

    #[test]
    fn test_builder_with_valid_inputs_builds_orchestrator() {
        let test_source = Box::new(TestHelper {});
        let test_destination = Box::new(TestHelper {});
        let _orchestrator = GenerateEmbeddingTask::builder()
            .source(test_source)
            .destination(test_destination)
            .embedding_client(Box::new(TestHelper {}))
            .chunk_size(2)
            .chunk_overlap(1)
            .build();
    }
}
