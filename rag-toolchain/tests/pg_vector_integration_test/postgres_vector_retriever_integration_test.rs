/// Postgres Integration Test
///
/// This test is an integration test for the postgres vector store and retriever.
///
/// It firstly tests by simply upserting vectors into the database and asserting that they are there
///
/// It firstly upserts the first two items of the test data into the vector databse. it is expected that when retrieving
/// based of the third item of the test data that the second item is returned. as that is the most similar text
///
/// Due to the nature of test containers we have to run each test all from the same function to allow them to all use the same
/// container.

#[cfg(all(test, feature = "pg_vector"))]
mod pg_vector {
    use lazy_static::lazy_static;
    use mockall::predicate::always;
    use mockall::*;
    use pgvector::Vector;
    use rag_toolchain::clients::{AsyncEmbeddingClient, OpenAIError};
    use rag_toolchain::common::{
        Chunk, Chunks, Embedding, OpenAIEmbeddingModel::TextEmbeddingAda002,
    };
    use rag_toolchain::retrievers::{
        AsyncRetriever, DistanceFunction, PostgresRetrieverError, PostgresVectorRetriever,
    };
    use rag_toolchain::stores::{EmbeddingStore, PostgresVectorStore};
    use serde_json::Value;
    use sqlx::postgres::PgPoolOptions;
    use sqlx::prelude::FromRow;
    use sqlx::{Pool, Postgres};
    use std::num::NonZeroU32;
    use testcontainers::{core::WaitFor, runners::AsyncRunner, GenericImage};

    const DISTANCE_FUNCTIONS: &[DistanceFunction] = &[
        DistanceFunction::Cosine,
        DistanceFunction::L2,
        DistanceFunction::InnerProduct,
    ];

    // We read some test data in, each chunk has some constant metadata just so
    // we can ensure that the metadata is being stored and retrieved correctly
    lazy_static! {
        static ref METADATA: Value = serde_json::json!({"test": "metadata"});
        static ref TEST_DATA: Vec<Embedding> = read_test_data();
    }

    fn get_image() -> GenericImage {
        GenericImage::new("ankane/pgvector", "latest")
            .with_wait_for(WaitFor::seconds(2))
            .with_wait_for(WaitFor::message_on_stdout(
                "database system is ready to accept connections",
            ))
            .with_env_var("POSTGRES_USER", "postgres")
            .with_env_var("POSTGRES_PASSWORD", "postgres")
            .with_env_var("POSTGRES_DB", "test_db")
            .with_exposed_port(5432)
    }

    fn set_env_vars(port: u16) {
        let host = format!("{}:{}", "localhost", port);
        std::env::set_var("POSTGRES_USER", "postgres");
        std::env::set_var("POSTGRES_PASSWORD", "postgres");
        std::env::set_var("POSTGRES_HOST", host);
        std::env::set_var("POSTGRES_DATABASE", "test_db");
    }

    #[tokio::test]
    async fn run_integration_tests() {
        let image = get_image();
        let container = image.start().await.unwrap();
        set_env_vars(container.get_host_port_ipv4(5432).await.unwrap());

        let connection_string = format!(
            "postgres://{}:{}@{}:{}/{}",
            "postgres",
            "postgres",
            "localhost",
            container.get_host_port_ipv4(5432).await.unwrap(),
            "test_db"
        );

        let pool: Pool<Postgres> = PgPoolOptions::new()
            .max_connections(1)
            .connect(&connection_string)
            .await
            .unwrap();

        sqlx::query("CREATE EXTENSION vector;")
            .execute(&pool)
            .await
            .unwrap();

        let case1 = test_store_persists_with_pool(pool);
        let case2 = test_batch_store_persists();
        let case3 = test_retriever_returns_correct_data();
        let case4 = test_retriever_with_embedding_client_error();

        let _ = tokio::join!(case1, case2, case3, case4);
    }

    async fn test_store_persists_with_pool(pool: Pool<Postgres>) {
        const TABLE_NAME: &str = "test_db_1";
        let embedding: Embedding = read_test_data()[0].clone();
        let pg_vector =
            PostgresVectorStore::try_new_with_pool(pool, TABLE_NAME, TextEmbeddingAda002)
                .await
                .unwrap();
        let _result = pg_vector
            .store(embedding.clone())
            .await
            .map_err(|_| panic!("panic"));
        assert_row(&pg_vector.get_pool(), 1, embedding, TABLE_NAME).await;
    }

    async fn test_batch_store_persists() {
        const TABLE_NAME: &str = "test_db_2";
        let pg_vector = PostgresVectorStore::try_new(TABLE_NAME, TextEmbeddingAda002)
            .await
            .unwrap();
        let _result = pg_vector
            .store_batch(TEST_DATA.clone())
            .await
            .map_err(|_| panic!("panic"));

        for (i, embedding) in TEST_DATA.iter().enumerate() {
            assert_row(
                &pg_vector.get_pool(),
                (i + 1) as i32,
                embedding.clone(),
                TABLE_NAME,
            )
            .await;
        }
    }

    async fn test_retriever_returns_correct_data() {
        const TABLE_NAME: &str = "test_db_3";
        let pg_vector = PostgresVectorStore::try_new(TABLE_NAME, TextEmbeddingAda002)
            .await
            .unwrap();
        let input: Vec<Embedding> = read_test_data();
        let data_to_store: &[Embedding] = &input[0..2];
        let _result = pg_vector
            .store_batch(data_to_store.to_vec())
            .await
            .map_err(|_| panic!("panic"));

        for (i, embedding) in data_to_store.iter().enumerate() {
            assert_row(
                &pg_vector.get_pool(),
                (i + 1) as i32,
                embedding.clone(),
                TABLE_NAME,
            )
            .await;
        }

        for distance_function in DISTANCE_FUNCTIONS {
            let test_data = TEST_DATA[2].clone();
            let mut mock_client: MockAsyncEmbeddingClient = MockAsyncEmbeddingClient::new();
            mock_client
                .expect_generate_embedding()
                .with(always())
                .returning(move |_| Ok(test_data.clone()));
            let retriever: PostgresVectorRetriever<MockAsyncEmbeddingClient> =
                pg_vector.as_retriever(mock_client, distance_function.clone());

            let result: Chunk = retriever
                .retrieve(
                    "This sentence is similar to a foo bar sentence .",
                    NonZeroU32::new(1).unwrap(),
                )
                .await
                .unwrap()
                .get(0)
                .unwrap()
                .to_owned();
            assert_eq!(result, *input[1].chunk());
        }
    }

    async fn test_retriever_with_embedding_client_error() {
        const TABLE_NAME: &str = "test_db_4";
        let pg_vector = PostgresVectorStore::try_new(TABLE_NAME, TextEmbeddingAda002)
            .await
            .unwrap();

        let mut mock_client: MockAsyncEmbeddingClient = MockAsyncEmbeddingClient::new();
        mock_client
            .expect_generate_embedding()
            .with(always())
            .returning(move |_| Err(OpenAIError::ErrorSendingRequest("error".to_string())));

        let retriever: PostgresVectorRetriever<MockAsyncEmbeddingClient> =
            pg_vector.as_retriever(mock_client, DistanceFunction::Cosine);

        let result = retriever
            .retrieve(
                "This sentence is similar to a foo bar sentence .",
                NonZeroU32::new(1).unwrap(),
            )
            .await
            .unwrap_err();
        assert!(matches!(
            result,
            PostgresRetrieverError::EmbeddingClientError(_)
        ));
    }

    async fn assert_row(
        pool: &Pool<Postgres>,
        id: i32,
        embedding: Embedding,
        table_name: &str,
    ) -> () {
        let row: RowData = query_row(pool, id, table_name).await;
        assert_eq!(row.id, id);
        assert_eq!(row.content, embedding.chunk().content());
        assert_eq!(row.embedding.to_vec(), embedding.vector());
        assert_eq!(row.metadata, *METADATA)
    }

    async fn query_row(pool: &Pool<Postgres>, id: i32, table_name: &str) -> RowData {
        let query: String = format!(
            "SELECT id, content, embedding, metadata FROM {} WHERE id = $1",
            table_name
        );

        sqlx::query_as::<_, RowData>(&query)
            .bind(id)
            .fetch_one(pool)
            .await
            .unwrap()
    }

    #[derive(FromRow)]
    struct RowData {
        id: i32,
        content: String,
        embedding: Vector,
        #[sqlx(json)]
        metadata: Value,
    }

    fn read_test_data() -> Vec<Embedding> {
        let file_string =
            std::fs::read_to_string("tests/pg_vector_integration_test/test-data.json").unwrap();
        let json: Vec<Value> = serde_json::from_str(&file_string).unwrap();
        let mut input_data: Vec<Embedding> = Vec::new();

        for object in json {
            let chunk: String = object["chunk"].to_string();
            let embedding: Vec<f32> = object["embedding"]
                .as_array()
                .unwrap()
                .into_iter()
                .map(|x| x.as_f64().unwrap() as f32)
                .collect();
            let chunk = Chunk::new_with_metadata(chunk, serde_json::json!({"test": "metadata"}));
            input_data.push(Embedding::new(chunk, embedding))
        }
        input_data
    }

    mock! {
        pub AsyncEmbeddingClient {}
        impl AsyncEmbeddingClient for AsyncEmbeddingClient {
            type ErrorType = OpenAIError;
            async fn generate_embedding(&self, text: Chunk) -> Result<Embedding, <Self as AsyncEmbeddingClient>::ErrorType>;
            async fn generate_embeddings(
                &self,
                text: Chunks,
            ) -> Result<Vec<Embedding>, <Self as AsyncEmbeddingClient>::ErrorType>;
        }
    }
}
