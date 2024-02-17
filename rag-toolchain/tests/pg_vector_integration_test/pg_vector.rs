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
    use async_trait::async_trait;
    use mockito::Mock;
    use pgvector::Vector;
    use rag_toolchain::clients::AsyncEmbeddingClient;
    use rag_toolchain::common::OpenAIEmbeddingModel;
    use rag_toolchain::common::{
        Chunk, Chunks, Embedding, OpenAIEmbeddingModel::TextEmbeddingAda002,
    };
    use rag_toolchain::retrievers::{AsyncRetriever, PostgresVectorRetriever};
    use rag_toolchain::stores::{
        DistanceFunction, EmbeddingStore, IndexTypes, NoIndex, PostgresVectorStore, HNSW, IVFFLAT,
    };
    use serde_json::Value;
    use sqlx::{postgres::PgRow, Pool, Postgres, Row};
    use std::fmt::format;
    use std::num::NonZeroU32;
    use testcontainers::{
        clients::Cli,
        core::{ExecCommand, WaitFor},
        GenericImage,
    };

    fn get_image() -> GenericImage {
        GenericImage::new("ankane/pgvector", "latest")
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
        let docker = Cli::default();
        let image = get_image();
        let container = docker.run(image);
        set_env_vars(container.get_host_port_ipv4(5432));

        let command_string =
            format!(r#"psql -U postgres -d test_db -c "CREATE EXTENSION IF NOT EXISTS vector;""#);
        let mut command: ExecCommand = ExecCommand::default();
        command.cmd = command_string;

        // Execute custom SQL commands to enable the extension
        let _output = container.exec(command);

        let case1 = run_hnsw_tests();
        let case2 = run_no_index_tests();
        //let case3 = test_retriever_returns_correct_data();

        let _ = tokio::join!(case1, case2);
    }

    async fn run_hnsw_tests() {
        let distance_functions: Vec<DistanceFunction> = vec![
            DistanceFunction::Cosine,
            DistanceFunction::InnerProduct,
            DistanceFunction::L2,
        ];
        for func in distance_functions.clone() {
            let table_name: String = format!("hnsw_table_1_{}", func);
            let hnsw: PostgresVectorStore<HNSW> = PostgresVectorStore::<HNSW>::try_new(
                &table_name,
                OpenAIEmbeddingModel::TextEmbeddingAda002,
                func,
            )
            .await
            .unwrap();
            test_store_persists("hnsw_table_1", hnsw).await;
        }

        for func in distance_functions.clone() {
            let table_name: String = format!("hnsw_table_2_{}", func);
            let hnsw: PostgresVectorStore<HNSW> = PostgresVectorStore::<HNSW>::try_new(
                &table_name,
                OpenAIEmbeddingModel::TextEmbeddingAda002,
                func,
            )
            .await
            .unwrap();
            test_batch_store_persists("hnsw_table_2", hnsw).await;
        }

        for func in distance_functions {
            let table_name: String = format!("hnsw_table_3_{}", func);
            let embedding_client = MockEmbeddingClient::new();
            let hnsw: PostgresVectorStore<HNSW> = PostgresVectorStore::<HNSW>::try_new(
                &table_name,
                OpenAIEmbeddingModel::TextEmbeddingAda002,
                func,
            )
            .await
            .unwrap();
            let retriever: PostgresVectorRetriever<MockEmbeddingClient> =
                hnsw.as_retriever(embedding_client);
            test_retriever_returns_correct_data(hnsw, retriever).await;
        }
    }

    async fn run_no_index_tests() {
        let distance_functions: Vec<DistanceFunction> = vec![
            DistanceFunction::Cosine,
            DistanceFunction::InnerProduct,
            DistanceFunction::L2,
        ];

        let table_name: String = format!("no_index_table_1");
        let no_index: PostgresVectorStore<NoIndex> = PostgresVectorStore::<NoIndex>::try_new(
            &table_name,
            OpenAIEmbeddingModel::TextEmbeddingAda002,
        )
        .await
        .unwrap();
        test_store_persists("hnsw_table_1", no_index).await;

        let table_name: String = format!("no_index_table_2");
        let no_index: PostgresVectorStore<NoIndex> = PostgresVectorStore::<NoIndex>::try_new(
            &table_name,
            OpenAIEmbeddingModel::TextEmbeddingAda002,
        )
        .await
        .unwrap();
        test_batch_store_persists(&table_name, no_index).await;

        for func in distance_functions {
            let table_name: String = format!("no_index_table_3_{}", func);
            let embedding_client = MockEmbeddingClient::new();
            let no_index: PostgresVectorStore<NoIndex> = PostgresVectorStore::<NoIndex>::try_new(
                &table_name,
                OpenAIEmbeddingModel::TextEmbeddingAda002,
            )
            .await
            .unwrap();
            let retriever: PostgresVectorRetriever<MockEmbeddingClient> =
                no_index.as_retriever(embedding_client, func);
            test_retriever_returns_correct_data(no_index, retriever).await;
        }
    }

    async fn test_store_persists<T>(table_name: &str, store: PostgresVectorStore<T>)
    where
        T: IndexTypes,
    {
        let (test_chunk, test_embedding): (Chunk, Embedding) = read_test_data()[0].clone();
        let _result = store
            .store((test_chunk.clone(), test_embedding.clone()))
            .await
            .map_err(|_| panic!("panic"));
        assert_row(
            &store.get_pool(),
            1,
            test_chunk.into(),
            test_embedding.into(),
            table_name,
        )
        .await;
    }

    async fn test_batch_store_persists<T>(table_name: &str, store: PostgresVectorStore<T>)
    where
        T: IndexTypes,
    {
        let input: Vec<(Chunk, Embedding)> = read_test_data();
        let _result = store
            .store_batch(input.clone())
            .await
            .map_err(|_| panic!("panic"));

        for (i, (chunk, embedding)) in input.iter().enumerate() {
            assert_row(
                &store.get_pool(),
                (i + 1) as i32,
                chunk.clone().into(),
                embedding.clone().into(),
                table_name,
            )
            .await;
        }
    }

    async fn test_retriever_returns_correct_data<T>(
        store: PostgresVectorStore<T>,
        retriever: PostgresVectorRetriever<MockEmbeddingClient>,
    ) where
        T: IndexTypes,
    {
        let input: Vec<(Chunk, Embedding)> = read_test_data();
        let data_to_store: Vec<(Chunk, Embedding)> = input[0..2].to_vec();
        let _result = store
            .store_batch(data_to_store.clone())
            .await
            .map_err(|_| panic!("panic"));

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
