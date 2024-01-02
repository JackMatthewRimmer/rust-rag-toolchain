#[cfg(all(test, feature = "pg_vector"))]
mod pg_vector {

    use async_trait::async_trait;
    use pgvector::Vector;
    use rag_toolchain::clients::traits::AsyncEmbeddingClient;
    use rag_toolchain::common::embedding_shared::OpenAIEmbeddingModel::TextEmbeddingAda002;
    use rag_toolchain::common::types::{Chunk, Chunks, Embedding};
    use rag_toolchain::retrievers::postgres_vector_retriever::PostgresVectorRetriever;
    use rag_toolchain::retrievers::traits::AsyncRetriever;
    use rag_toolchain::stores::postgres_vector_store::PostgresVectorStore;
    use rag_toolchain::stores::traits::EmbeddingStore;
    use serde_json::Value;
    use sqlx::postgres::PgRow;
    use sqlx::{Pool, Postgres, Row};

    #[tokio::test]
    async fn test_store_persists() {
        const TABLE_NAME: &str = "test_db_1";
        std::env::set_var("POSTGRES_USER", "postgres");
        std::env::set_var("POSTGRES_PASSWORD", "postgres");
        std::env::set_var("POSTGRES_HOST", "localhost");
        std::env::set_var("POSTGRES_DATABASE", "pg_vector");

        let (test_chunk, test_embedding): (Chunk, Embedding) = read_test_data()[0].clone();

        let pg_vector = PostgresVectorStore::new(TABLE_NAME, TextEmbeddingAda002)
            .await
            .unwrap();
        let _result = pg_vector
            .store((test_chunk.clone(), test_embedding.clone()))
            .await
            .map_err(|_| panic!("panic"));
        assert_row(
            &pg_vector.pool,
            1,
            test_chunk.into(),
            test_embedding.into(),
            TABLE_NAME,
        )
        .await;
    }

    #[tokio::test]
    async fn test_batch_store_persists() {
        const TABLE_NAME: &str = "test_db_2";
        std::env::set_var("POSTGRES_USER", "postgres");
        std::env::set_var("POSTGRES_PASSWORD", "postgres");
        std::env::set_var("POSTGRES_HOST", "localhost");
        std::env::set_var("POSTGRES_DATABASE", "pg_vector");
        let pg_vector = PostgresVectorStore::new(TABLE_NAME, TextEmbeddingAda002)
            .await
            .unwrap();
        let input: Vec<(Chunk, Embedding)> = read_test_data();
        let _result = pg_vector
            .store_batch(input.clone())
            .await
            .map_err(|_| panic!("panic"));

        for (i, (chunk, embedding)) in input.iter().enumerate() {
            assert_row(
                &pg_vector.pool,
                (i + 1) as i32,
                chunk.clone().into(),
                embedding.clone().into(),
                TABLE_NAME,
            )
            .await;
        }
    }

    #[tokio::test]
    async fn test_retriever_returns_correct_data() {
        const TABLE_NAME: &str = "test_db_3";
        std::env::set_var("POSTGRES_USER", "postgres");
        std::env::set_var("POSTGRES_PASSWORD", "postgres");
        std::env::set_var("POSTGRES_HOST", "localhost");
        std::env::set_var("POSTGRES_DATABASE", "pg_vector");
        let pg_vector = PostgresVectorStore::new(TABLE_NAME, TextEmbeddingAda002)
            .await
            .unwrap();
        let input: Vec<(Chunk, Embedding)> = read_test_data();
        let data_to_store : Vec<(Chunk, Embedding)>= input[0..1].to_vec();
        let _result = pg_vector
            .store_batch(data_to_store.clone())
            .await
            .map_err(|_| panic!("panic"));

        for (i, (chunk, embedding)) in data_to_store.iter().enumerate() {
            assert_row(
                &pg_vector.pool,
                (i + 1) as i32,
                chunk.clone().into(),
                embedding.clone().into(),
                TABLE_NAME,
            )
            .await;
        }

        let mock_client: MockEmbeddingClient = MockEmbeddingClient::new();
        let retriever: PostgresVectorRetriever<MockEmbeddingClient> =
            pg_vector.as_retriever(mock_client);

        let result: Chunk = retriever
            .retrieve("Ghosts are really dangerous and scary, Snakes are really dangerous and scary")
            .await
            .unwrap();

        assert_eq!(result, input[1].0);
    }

    async fn assert_row(
        pool: &Pool<Postgres>,
        id: i32,
        text: String,
        embeddings: Vec<f32>,
        table_name: &str,
    ) -> () {
        let row: RowData = query_row(pool, id, table_name).await;
        assert_eq!(row.id, id);
        assert_eq!(row.content, text);
        assert_eq!(row.embedding, embeddings);
    }

    async fn query_row(pool: &Pool<Postgres>, id: i32, table_name: &str) -> RowData {
        let query: String = format!(
            "SELECT id, content, embedding FROM {} WHERE id = $1",
            table_name
        );

        let query: PgRow = sqlx::query(&query).bind(id).fetch_one(pool).await.unwrap();
        RowData {
            id: query.get::<i32, _>("id"),
            content: query.get::<String, _>("content"),
            embedding: query.get::<Vector, _>("embedding").to_vec(),
        }
    }

    struct RowData {
        id: i32,
        content: String,
        embedding: Vec<f32>,
    }

    fn read_test_data() -> Vec<(Chunk, Embedding)> {
        let file_string =
            std::fs::read_to_string("tests/pg_vector_integration_test/test-data.json").unwrap();
        let json: Vec<Value> = serde_json::from_str(&file_string).unwrap();
        let mut input_data: Vec<(Chunk, Embedding)> = Vec::new();

        for object in json {
            let chunk: String = object["chunk"].to_string();
            let embedding: Vec<f32> = object["embedding"]
                .as_array()
                .unwrap()
                .into_iter()
                .map(|x| x.as_f64().unwrap() as f32)
                .collect();
            input_data.push((Chunk::from(chunk), Embedding::from(embedding)))
        }
        input_data
    }

    pub struct MockEmbeddingClient {
        input_data: Vec<(Chunk, Embedding)>,
    }
    impl MockEmbeddingClient {
        pub fn new() -> Self {
            MockEmbeddingClient {
                input_data: read_test_data(),
            }
        }
    }
    #[async_trait]
    impl AsyncEmbeddingClient for MockEmbeddingClient {
        type ErrorType = std::io::Error;
        async fn generate_embedding(
            &self,
            _text: Chunk,
        ) -> Result<(Chunk, Embedding), Self::ErrorType> {
            Ok(self.input_data[2].clone())
        }
        async fn generate_embeddings(
            &self,
            _text: Chunks,
        ) -> Result<Vec<(Chunk, Embedding)>, Self::ErrorType> {
            unimplemented!()
        }
    }
}
