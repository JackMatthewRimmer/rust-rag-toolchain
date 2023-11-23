use dotenv_codegen::dotenv;

/// # PgVector
///
/// This struct is used to store and retriever information needed to connect to a Postgres database
/// and should be passed to an embedding task as a destination for the data to be stored.
///
/// If a table already exists with the same name, the table will be dropped and recreated.
///
/// # Required Environment Variables
///
/// * POSTGRES_USERNAME: The username to connect to the database with
/// * POSTGRES_PASSWORD: The password to connect to the database with
/// * POSTGRES_HOST: The host to connect to the database with
///
/// Place these variables in a .env file in the root of your project.
/// If these variables are not set you cannot compile ```PgVector::new("table_name")```
///
/// # Output table format
/// Columns: | id (int) | content (text) | embedding (vector) |
pub struct PgVector {
    db_name: String,
    username: &'static str,
    password: &'static str,
    host: &'static str,
}

impl PgVector {
    /// # Note
    /// [`PgVector::new`] will not compile if the required environment
    /// variables are not set
    ///
    /// # Arguments
    /// * `db_name` - The name of the table to store the embeddings in.
    ///
    /// # Returns
    /// the constructed [`PgVector`] struct
    pub fn new(db_name: impl Into<String>) -> PgVector {
        // We verify that the environment variables are set at compile time
        let username: &str = dotenv!("POSTGRES_USERNAME");
        let password: &str = dotenv!("POSTGRES_PASSWORD");
        let host: &str = dotenv!("POSTGRES_HOST");
        let db_name: String = db_name.into();

        PgVector {
            db_name,
            username,
            password,
            host,
        }
    }
}
