use dotenv::dotenv;
/// # PgVector
///
/// This struct is used to store the information needed to connect to a Postgres database
/// and should be passed to an embedding task as a destination for the data to be stored.
///
/// If a table already exists with the same name, a new one will be generated in order to
/// preserve the embeddings as these cost money.
///
/// # Required Environment Variables
///
/// * POSTGRES_USERNAME
/// * POSTGRES_PASSWORD
/// * POSTGRES_HOST
///
/// Place these variables in a .env file in the root of your project.
///
/// # Output table format
/// name by default is embeddings_uuid if not specified.
/// Columns: | id (int) | content (text) | embedding (vector) |
pub struct PgVector {
    db_name: String,
    username: String,
    password: String,
    host: String,
}

impl PgVector {
    /// # Arguments
    /// * `db_name` - The name of the table to store the embeddings in.
    /// Should be None if you want a default table name
    ///
    /// # Returns
    /// `Result<PgVector, dotenv::Error>` - Will return an error if the
    /// appropriate env vars are not set
    pub fn new(db_name: Option<String>) -> Result<PgVector, dotenv::Error> {
        dotenv().ok();
        let username: String = dotenv::var("POSTGRES_USERNAME")?;
        let password: String = dotenv::var("POSTGRES_PASSWORD")?;
        let host: String = dotenv::var("POSTGRES_HOST")?;
        Ok(PgVector {
            db_name: match db_name {
                Some(name) => name,
                None => format!("embeddings_{}", uuid::Uuid::new_v4()),
            },
            username,
            password,
            host,
        })
    }
}
