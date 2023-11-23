use std::env;

use dotenv::dotenv;

pub struct EnvVarError(String);

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
///
/// # Output table format
/// Columns: | id (int) | content (text) | embedding (vector) |
pub struct PgVector {
    db_name: String,
    username: String,
    password: String,
    host: String,
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
    pub fn new(db_name: impl Into<String>) -> Result<PgVector, EnvVarError> {
        dotenv().ok();
        let username: String = match env::var("POSTGRES_USERNAME") {
            Ok(username) => username,
            Err(_) => return Err(EnvVarError("Error: POSTGRES_USERNAME not set".into())),
        };
        let password: String = match env::var("POSTGRES_PASSWORD") {
            Ok(password) => password,
            Err(_) => return Err(EnvVarError("Error: POSTGRES_PASSWORD not set".into())),
        };
        let host: String = match env::var("POSTGRES_HOST") {
            Ok(host) => host,
            Err(_) => return Err(EnvVarError("Error: POSTGRES_HOST not set".into())),
        };
        let db_name: String = db_name.into();

        Ok(PgVector {
            db_name,
            username,
            password,
            host,
        })
    }
}
