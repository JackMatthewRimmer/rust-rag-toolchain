/*
What needs to be done to make this integration test fully functional:
We need to create a mocked client that returns some form of real vector data for two different strings
We then use that mocked client to create a PostgresVectorStore and store the two strings
We then convert the store to a retrieve something similar to each string
and we would expect back the string that had a certain vector
*/

#[cfg(all(test, feature = "pg_vector"))]
mod pg_vector {

    use pgvector::Vector;
    use rag_toolchain::common::embedding_shared::OpenAIEmbeddingModel::TextEmbeddingAda002;
    use rag_toolchain::common::types::{Chunk, Embedding};
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
            std::fs::read_to_string("tests/pg_vector_integration_test/test_data.json").unwrap();
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
}
